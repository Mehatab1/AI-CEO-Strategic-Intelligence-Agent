"""
Shared helpers for the SAP Strategic Intelligence agentic pipeline.

This module implements the explicit pipeline requested in the course
clarification:

    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate

Each stage is a separate, logged step rather than something left implicit
inside one long tool-calling loop:

  - PLAN     : a forced tool call where the model states its goal in its own
               words and lists the concrete steps it intends to take, BEFORE
               touching any data. (Planning before execution.)
  - RETRIEVE : an open-ended loop where the model decides which retrieval/
               memory tools to call, with what arguments, and how many
               rounds it needs, signalling completion itself.
               (Autonomous decision-making, tool usage, memory, multi-step
               execution.)
  - ANALYZE  : a forced tool call where the model must summarise what the
               gathered evidence actually shows, before deciding anything.
  - DECIDE + RECOMMEND : a forced tool call producing the structured
               recommendation(s), grounded in the plan + evidence + analysis.
  - VALIDATE : a second, separate forced tool call that checks the draft
               recommendation(s) against the evidence actually gathered,
               and can confirm, revise, or reject each one before anything
               is saved or shown to a user.

Every stage's output is captured so the full pipeline can be inspected
after the fact - this is the artifact that makes "the system planned",
"the system validated its answer" etc. demonstrable rather than assumed.
"""
import ast
import json
import os
import time
from pathlib import Path

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))  # local CPU models are slow, give it time

MAX_RETRIEVAL_ITERATIONS_DEFAULT = 6
MAX_SINGLE_CALL_RETRIES_DEFAULT = 3


# ----------------------------------------------------------------------------
# Low-level Ollama plumbing
# ----------------------------------------------------------------------------

def find_data_dir():
    """Find the data folder whether we run from agents/ or repo root."""
    cwd = Path.cwd()
    return next(
        (p for p in [cwd / "notebook" / "data", cwd.parent / "notebook" / "data"] if p.exists()),
        cwd,
    )


def ask_llm_raw(prompt):
    """Single-shot completion - kept around only for the warmup call."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def warmup():
    print("warming up model...")
    start = time.time()
    ask_llm_raw("hi")
    print("warmup done in", round(time.time() - start, 1), "sec")


def _call_ollama_chat(messages, tools):
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_arguments(raw_args):
    """Ollama normally returns tool-call arguments as a dict already, but be
    defensive in case a model emits them as a JSON string instead."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return {}


def load_previous_output(path: Path):
    """Memory: read this agent's own output from its last run, if any."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ----------------------------------------------------------------------------
# Generic single-decision primitive, used by PLAN / ANALYZE / DECIDE / VALIDATE
# ----------------------------------------------------------------------------

def _unwrap_tool_envelope(args):
    """
    Defends against a model occasionally emitting its *entire* tool-call
    envelope ({"name": ..., "parameters": {...}}) as text content buried
    inside the arguments, instead of returning the real arguments
    directly - e.g. {"planned_steps": ['{"name": "submit_plan",
    "parameters": {"goal": "...", "planned_steps": [...]}}']} instead of
    just {"goal": "...", "planned_steps": [...]}.

    Checks two shapes: the envelope as the whole args dict itself, or
    hiding as a JSON string inside one of the values (commonly the sole
    item of a list-valued field). Returns the real arguments if an
    envelope is found and unwrapped, otherwise returns args unchanged.
    """
    def _try_parse_envelope(value):
        if not isinstance(value, str):
            return None
        try:
            parsed = json.loads(value.strip())
        except (json.JSONDecodeError, AttributeError):
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("parameters"), dict):
            return parsed["parameters"]
        return None

    if not isinstance(args, dict):
        return args

    # Case 1: the envelope is the whole args dict itself.
    if "name" in args and isinstance(args.get("parameters"), dict):
        return args["parameters"]

    # Case 2: the envelope is hiding as a string inside one of the values.
    for value in args.values():
        if isinstance(value, list) and len(value) == 1:
            unwrapped = _try_parse_envelope(value[0])
            if unwrapped:
                return unwrapped
        unwrapped = _try_parse_envelope(value)
        if unwrapped:
            return unwrapped

    return args


def run_single_tool_call(system_prompt, user_message, tool_schema, tool_name,
                          max_attempts=MAX_SINGLE_CALL_RETRIES_DEFAULT, verbose=True):
    """
    Forces a single decision point: the model has exactly one tool available
    and must call it to produce structured output for that stage.

    Returns (args, fallback_text):
      args          - the tool's arguments dict if the model called it.
      fallback_text - free text if a weaker model answered without using
                       the tool (args will be None in that case). Callers
                       should treat this as a degraded-but-non-fatal result.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(max_attempts):
        result = _call_ollama_chat(messages, [tool_schema])
        message = result.get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {})
                if fn.get("name") == tool_name:
                    args = _parse_arguments(fn.get("arguments"))
                    args = _unwrap_tool_envelope(args)
                    return args, None
            # Model called something else / malformed name - nudge and retry.
            messages.append(message)
            messages.append({
                "role": "user",
                "content": f"Please call the `{tool_name}` tool specifically with your answer.",
            })
            continue

        # No tool call at all - free text fallback.
        return None, message.get("content", "")

    if verbose:
        print(f"  (gave up after {max_attempts} attempts waiting for `{tool_name}`)")
    return None, None


# ----------------------------------------------------------------------------
# STAGE 1: PLAN
# ----------------------------------------------------------------------------

PLAN_TOOL_NAME = "submit_plan"
PLAN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": PLAN_TOOL_NAME,
        "description": (
            "Submit your plan before doing anything else: restate the goal "
            "in your own words and list the concrete steps you intend to "
            "take to achieve it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The goal, restated in your own words"},
                "planned_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 to 5 concrete steps - e.g. specific queries or data sources you intend to check",
                },
            },
            "required": ["goal", "planned_steps"],
        },
    },
}


def run_plan_stage(goal_description, persona_context="",
                    max_attempts=MAX_SINGLE_CALL_RETRIES_DEFAULT, verbose=True):
    """STAGE 1 - PLAN. Returns {"goal": str, "planned_steps": [str, ...]}.

    persona_context, if given, is prepended to the system prompt and kept
    consistent across every other stage too (see the same parameter on
    run_retrieval_stage/run_analyze_stage/run_decide_stage/run_validate_stage).
    It exists because relying on the model to faithfully carry a persona or
    perspective forward through its own restated "goal" text is not
    reliable - it can drift (e.g. start advising a competitor mentioned in
    the question instead of the entity it's actually working for). Injecting
    it directly into every stage's prompt removes that dependency.
    """
    prefix = f"{persona_context.strip()}\n\n" if persona_context.strip() else ""
    system_prompt = (
        prefix
        + "You are about to carry out an analysis task. Before doing anything "
        "else, state your plan by calling submit_plan: restate the goal in "
        "your own words, and list 3 to 5 concrete steps you intend to take "
        "(e.g. specific search queries, or which data sources you'll check "
        "first) to achieve it. Do not investigate anything yet - just plan."
    )
    user_message = f"Goal: {goal_description}"

    args, fallback_text = run_single_tool_call(
        system_prompt, user_message, PLAN_TOOL_SCHEMA, PLAN_TOOL_NAME, max_attempts, verbose
    )

    if args:
        plan = {"goal": args.get("goal", goal_description), "planned_steps": args.get("planned_steps", [])}
    else:
        plan = {
            "goal": goal_description,
            "planned_steps": [fallback_text] if fallback_text else ["No explicit plan returned by the model."],
        }

    if verbose:
        print("PLAN:", json.dumps(plan, indent=2, default=str))
    return plan


# ----------------------------------------------------------------------------
# STAGE 2: RETRIEVE
# ----------------------------------------------------------------------------

FINISHED_RETRIEVING_TOOL_NAME = "finished_retrieving"
_FINISHED_RETRIEVING_SCHEMA = {
    "type": "function",
    "function": {
        "name": FINISHED_RETRIEVING_TOOL_NAME,
        "description": "Call this once you have gathered enough evidence and are ready to move to analysis.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def run_retrieval_stage(plan, tool_schemas, tool_handlers, persona_context="",
                         max_iterations=MAX_RETRIEVAL_ITERATIONS_DEFAULT, verbose=True,
                         initial_user_message=None):
    """
    STAGE 2 - RETRIEVE. The agent decides which of `tool_schemas` to call,
    with what arguments, and how many rounds it needs - then signals it's
    done by calling `finished_retrieving`. Returns a trace list of
    {"step", "tool", "arguments", "result"} dicts - this trace is the
    evidence base for the Analyze/Decide/Validate stages that follow.

    initial_user_message: optional override for the opening user turn.
    When callers can infer which tools are most relevant from the plan
    (e.g. a competitor question → suggest run_competitor_intelligence),
    passing a targeted hint here dramatically reduces first-step tool-call
    failures on smaller local models.
    """
    full_schemas = list(tool_schemas) + [_FINISHED_RETRIEVING_SCHEMA]
    prefix = f"{persona_context.strip()}\n\n" if persona_context.strip() else ""

    tool_names = [s.get("function", {}).get("name", "") for s in tool_schemas]

    messages = [
        {
            "role": "system",
            "content": (
                prefix
                + "You are executing the Retrieve stage of an analysis pipeline.\n"
                f"Your plan was:\n{json.dumps(plan, default=str)}\n\n"
                "You MUST call at least one retrieval tool. "
                "Call your retrieval tools as many times as needed to gather "
                "real evidence relevant to the plan above. You decide the "
                "queries and how many calls are enough. When you have enough "
                f"evidence, call `{FINISHED_RETRIEVING_TOOL_NAME}` to move on. "
                "Do not draw conclusions yet - just gather evidence.\n"
                f"Available tools: {', '.join(tool_names)}."
            ),
        },
        {
            "role": "user",
            "content": initial_user_message or "Begin retrieving evidence now.",
        },
    ]

    trace = []
    nudged = False  # only nudge once per retrieval session

    for step in range(max_iterations):
        result = _call_ollama_chat(messages, full_schemas)
        message = result.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # If no tool was called at all yet, send one strong nudge before giving up.
            if not trace and not nudged:
                nudged = True
                messages.append(message)
                messages.append({
                    "role": "user",
                    "content": (
                        "You have not called any retrieval tool yet, but this stage requires "
                        "real evidence. You MUST call at least one tool before finishing. "
                        "Look at your plan and call the most relevant tool right now. "
                        f"Do not call `{FINISHED_RETRIEVING_TOOL_NAME}` until you have "
                        "retrieved some evidence first."
                    ),
                })
                if verbose:
                    print(f"  [retrieve step {step + 1}] no tool call - sending nudge")
                continue

            if verbose:
                print(f"  [retrieve step {step + 1}] no tool call - treating retrieval as complete.")
            break

        messages.append(message)
        done = False

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args = _parse_arguments(fn.get("arguments"))

            if name == FINISHED_RETRIEVING_TOOL_NAME:
                done = True
                trace.append({"step": step + 1, "tool": name, "arguments": args})
                if verbose:
                    print(f"  [retrieve step {step + 1}] agent signalled done retrieving")
                continue

            handler = tool_handlers.get(name)
            if handler is None:
                tool_output = f"Unknown tool: {name}"
            else:
                try:
                    tool_output = handler(args)
                except Exception as e:
                    tool_output = f"Tool error: {str(e)}"

            trace.append({"step": step + 1, "tool": name, "arguments": args, "result": tool_output})
            if verbose:
                preview = json.dumps(args, default=str)[:160]
                print(f"  [retrieve step {step + 1}] called `{name}` with {preview}")

            messages.append({"role": "tool", "content": json.dumps(tool_output, default=str)})

        if done:
            break

    return trace


def build_evidence_text(retrieval_trace, max_chars=6000):
    """Concatenate the actual tool outputs gathered in Stage 2 into a single
    text blob, so later stages can be grounded against what was really
    retrieved rather than the model's unconstrained memory."""
    parts = []
    for entry in retrieval_trace:
        if "result" not in entry:
            continue
        header = f"[{entry.get('tool')} | args={json.dumps(entry.get('arguments'), default=str)}]"
        parts.append(f"{header}\n{json.dumps(entry['result'], default=str)}")
    text = "\n\n".join(parts) if parts else "(no evidence was retrieved)"
    return text[:max_chars]


# ----------------------------------------------------------------------------
# STAGE 3: ANALYZE
# ----------------------------------------------------------------------------

ANALYZE_TOOL_NAME = "submit_analysis"


def make_analyze_tool_schema(item_noun="findings"):
    return {
        "type": "function",
        "function": {
            "name": ANALYZE_TOOL_NAME,
            "description": (
                f"Submit your analysis of the gathered evidence, before "
                f"deciding on final {item_noun}. Identify the key patterns "
                f"or observations in the evidence and which candidate "
                f"{item_noun} they support."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3 to 6 short observations drawn directly from the evidence above",
                    },
                    "candidate_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"short working titles of candidate {item_noun} suggested by the observations",
                    },
                },
                "required": ["key_observations", "candidate_items"],
            },
        },
    }


def run_analyze_stage(plan, evidence_text, tool_schema, persona_context="",
                       max_attempts=MAX_SINGLE_CALL_RETRIES_DEFAULT, verbose=True):
    """STAGE 3 - ANALYZE. Returns {"key_observations": [...], "candidate_items": [...]}."""
    tool_name = tool_schema["function"]["name"]
    prefix = f"{persona_context.strip()}\n\n" if persona_context.strip() else ""
    system_prompt = (
        prefix
        + "You are executing the Analyze stage of an analysis pipeline.\n\n"
        f"Plan:\n{json.dumps(plan, default=str)}\n\n"
        f"Evidence gathered during Retrieve:\n{evidence_text}\n\n"
        f"Call `{tool_name}` with your analysis of this evidence before any "
        "decision is made. Base your analysis only on the evidence above."
    )

    args, fallback_text = run_single_tool_call(
        system_prompt, "Analyze the evidence now.", tool_schema, tool_name, max_attempts, verbose
    )

    if args:
        analysis = args
    else:
        analysis = {
            "key_observations": [fallback_text] if fallback_text else ["No explicit analysis returned by the model."],
            "candidate_items": [],
        }

    if verbose:
        print("ANALYSIS:", json.dumps(analysis, indent=2, default=str))
    return analysis


# ----------------------------------------------------------------------------
# STAGE 4: DECIDE + RECOMMEND
# ----------------------------------------------------------------------------

def run_decide_stage(plan, evidence_text, analysis, tool_schema, tool_name, persona_context="",
                      max_attempts=MAX_SINGLE_CALL_RETRIES_DEFAULT, verbose=True):
    """STAGE 4 - DECIDE + RECOMMEND. Returns the tool's argument dict, or {}."""
    prefix = f"{persona_context.strip()}\n\n" if persona_context.strip() else ""
    system_prompt = (
        prefix
        + "You are executing the Decide and Recommend stage of an analysis "
        "pipeline.\n\n"
        f"Plan:\n{json.dumps(plan, default=str)}\n\n"
        f"Evidence gathered:\n{evidence_text}\n\n"
        f"Your analysis of that evidence:\n{json.dumps(analysis, default=str)}\n\n"
        f"Based on all of the above, call `{tool_name}` exactly once with "
        "your final structured recommendation(s). Every recommendation must "
        "be traceable to the evidence above - do not invent anything."
    )

    args, fallback_text = run_single_tool_call(
        system_prompt, "Decide and submit your recommendations now.",
        tool_schema, tool_name, max_attempts, verbose
    )

    if verbose:
        print("DRAFT DECISION:", json.dumps(args, indent=2, default=str) if args else f"(none - fallback text: {fallback_text})")
    return args or {}


# ----------------------------------------------------------------------------
# STAGE 5: VALIDATE
# ----------------------------------------------------------------------------

def run_validate_stage(draft_payload, evidence_text, tool_schema, tool_name, persona_context="",
                        max_attempts=MAX_SINGLE_CALL_RETRIES_DEFAULT, verbose=True):
    """
    STAGE 5 - VALIDATE. Separate, second pass: checks the draft
    recommendation(s) against the evidence actually retrieved, and can
    confirm, revise, or reject each one. Returns the validated payload, or
    None if the model never produced one (caller should fall back to the
    unvalidated draft and mark it as such).
    """
    prefix = f"{persona_context.strip()}\n\n" if persona_context.strip() else ""
    system_prompt = (
        prefix
        + "You are executing the Validate stage of an analysis pipeline. "
        "Below are draft recommendation(s) and the evidence they should be "
        "grounded in. For each draft item, check whether the evidence "
        "actually supports it. Mark it Supported, Revised (if you adjusted "
        "it to better match the evidence), or Unsupported (if the evidence "
        "does not back it up) - do not invent new evidence to justify a "
        "weak item.\n\n"
        f"Draft recommendation(s):\n{json.dumps(draft_payload, default=str)}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Call `{tool_name}` exactly once with your validated result."
    )

    args, fallback_text = run_single_tool_call(
        system_prompt, "Validate the recommendation(s) now.",
        tool_schema, tool_name, max_attempts, verbose
    )

    if verbose:
        print("VALIDATED:", json.dumps(args, indent=2, default=str) if args else f"(none - fallback text: {fallback_text})")
    return args


def _parse_structured_string(s):
    """
    Try to recover real structure (a list or dict) from a string the model
    produced instead of real nested objects in a tool call. Three failure
    modes observed in practice, tried in order:

      1. Proper JSON (double-quoted) - just json.loads.
      2. Python dict/list repr (single-quoted, e.g. {'key': 'value'}) -
         json.loads can't parse this; ast.literal_eval can.
      3. Truncated mid-generation (the model ran out of tokens partway
         through the last item) - neither parser handles a dangling,
         unclosed string/object. Salvage by trimming back to the last
         complete '}' and closing the list there, recovering every item
         that did finish generating rather than losing all of them.

    Returns the parsed list/dict, or None if nothing could be recovered.
    """
    stripped = s.strip()
    if not (stripped.startswith("[") or stripped.startswith("{")):
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            continue

    if stripped.startswith("["):
        positions = [i for i, c in enumerate(stripped) if c == "}"]
        for pos in reversed(positions):
            candidate = stripped[: pos + 1] + "]"
            for parser in (json.loads, ast.literal_eval):
                try:
                    result = parser(candidate)
                    if isinstance(result, list):
                        return result
                except Exception:
                    continue

    return None


def normalize_items(items, string_key="title"):
    """
    Defensively normalize what should be a list of item-objects, but where
    the model may not have followed the schema at all - e.g. it returned a
    single plain string or a single dict instead of a list of objects.

    This matters more than it looks: a bare string passed to a naive
    `for item in items` loop iterates CHARACTER BY CHARACTER, silently
    turning one malformed field into hundreds of fake single-character
    "items" - which is exactly what produced the 669-item bug. Guard the
    container shape first, then normalize each individual item.
    """
    if items is None:
        return []
    if isinstance(items, str):
        # The model may have returned the entire array as one string
        # instead of a real list - try to recover real structure (see
        # _parse_structured_string for the failure modes handled) before
        # falling back to treating it as one bare title.
        parsed = _parse_structured_string(items)
        if isinstance(parsed, list):
            return normalize_items(parsed, string_key)
        if isinstance(parsed, dict):
            return [parsed]
        return [{string_key: items}]
    if isinstance(items, dict):
        # The model returned one object instead of a list of objects.
        return [items]

    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            # Same double-encoding problem, one level down: an individual
            # list item is itself a string containing the real structure.
            parsed = _parse_structured_string(item)
            if isinstance(parsed, list):
                normalized.extend(normalize_items(parsed, string_key))
            elif isinstance(parsed, dict):
                normalized.append(parsed)
            else:
                normalized.append({string_key: item})
        else:
            normalized.append({string_key: str(item)})
    return normalized


def attach_default_validation(items, status="Unverified",
                                note="Validate stage did not return structured output - showing unvalidated draft."):
    """Fallback used when the Validate stage fails entirely: stamp every
    draft item as Unverified rather than silently presenting it as validated."""
    for item in items:
        if isinstance(item, dict):
            item.setdefault("validation_status", status)
            item.setdefault("validation_notes", note)
    return items


# ----------------------------------------------------------------------------
# Full pipeline convenience wrapper (for the list-of-items specialist agents)
# ----------------------------------------------------------------------------

def run_full_pipeline(
    goal_description,
    retrieval_tool_schemas,
    retrieval_tool_handlers,
    analyze_tool_schema,
    decide_tool_schema,
    decide_tool_name,
    decide_items_key,
    validate_tool_schema,
    validate_tool_name,
    validate_items_key,
    persona_context="",
    max_retrieval_iterations=MAX_RETRIEVAL_ITERATIONS_DEFAULT,
    verbose=True,
):
    """
    Runs the full Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend
    -> Validate pipeline for agents whose Decide/Validate tools return a
    list of items under a known key (e.g. {"opportunities": [...]}).

    persona_context is optional and defaults to "" (no-op) - the four
    single-domain specialist agents (opportunity/risk/trend/competitor)
    don't need it since their domain is already unambiguous. It exists for
    callers like the interactive CEO chat, where the question itself can
    name a competitor prominently and the model needs a standing reminder
    of whose perspective it's actually working from.

    Returns (validated_items, pipeline_dict) where pipeline_dict captures
    every stage's output for later inspection / saving.
    """
    if verbose:
        print("\n" + "=" * 70 + "\nSTAGE 1: PLAN\n" + "=" * 70)
    plan = run_plan_stage(goal_description, persona_context=persona_context, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70 + "\nSTAGE 2: RETRIEVE\n" + "=" * 70)
    retrieval_trace = run_retrieval_stage(
        plan, retrieval_tool_schemas, retrieval_tool_handlers,
        persona_context=persona_context, max_iterations=max_retrieval_iterations, verbose=verbose,
    )
    evidence_text = build_evidence_text(retrieval_trace)

    if verbose:
        print("\n" + "=" * 70 + "\nSTAGE 3: ANALYZE\n" + "=" * 70)
    analysis = run_analyze_stage(plan, evidence_text, analyze_tool_schema, persona_context=persona_context, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70 + "\nSTAGE 4: DECIDE + RECOMMEND\n" + "=" * 70)
    draft = run_decide_stage(
        plan, evidence_text, analysis, decide_tool_schema, decide_tool_name,
        persona_context=persona_context, verbose=verbose,
    )
    draft_items = normalize_items(draft.get(decide_items_key, []))

    if verbose:
        print("\n" + "=" * 70 + "\nSTAGE 5: VALIDATE\n" + "=" * 70)
    validated = run_validate_stage(
        {decide_items_key: draft_items}, evidence_text, validate_tool_schema, validate_tool_name,
        persona_context=persona_context, verbose=verbose,
    )
    if validated and validated.get(validate_items_key):
        validated_items = normalize_items(validated[validate_items_key])
    else:
        validated_items = attach_default_validation(list(draft_items))

    pipeline = {
        "goal": goal_description,
        "plan": plan,
        "retrieval_trace": retrieval_trace,
        "analysis": analysis,
        "draft_recommendations": draft_items,
        "validated_recommendations": validated_items,
    }
    return validated_items, pipeline


# ----------------------------------------------------------------------------
# Saving helpers
# ----------------------------------------------------------------------------

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print("saved to", path)