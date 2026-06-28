import os
import time
import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

print(f"Testing model: {MODEL}")
print("=" * 60)

# Test 1: plain chat, no tools - baseline inference speed
print("\n[Test 1] Plain chat (no tools)...")
start = time.time()
try:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "stream": False,
        },
        timeout=180,
    )
    elapsed = time.time() - start
    print(f"  status: {resp.status_code}")
    print(f"  time: {elapsed:.1f}s")
    print(f"  response: {resp.json().get('message', {}).get('content', '')[:200]}")
except Exception as e:
    elapsed = time.time() - start
    print(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")

# Test 2: chat with a tool available - does tool-calling specifically hang?
print("\n[Test 2] Chat with a tool available...")
start = time.time()
try:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What's 2+2? Use the test tool if you need to."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test",
                        "description": "A test tool that does nothing useful.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": False,
        },
        timeout=180,
    )
    elapsed = time.time() - start
    data = resp.json()
    tool_calls = data.get("message", {}).get("tool_calls")
    print(f"  status: {resp.status_code}")
    print(f"  time: {elapsed:.1f}s")
    print(f"  tool_calls field present: {tool_calls is not None and tool_calls != []}")
    print(f"  tool_calls value: {tool_calls}")
    print(f"  content (if any): {data.get('message', {}).get('content', '')[:300]}")
except Exception as e:
    elapsed = time.time() - start
    print(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Done.")