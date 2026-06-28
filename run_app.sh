#!/bin/bash
chmod +x cloudflared-linux-amd64
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 &
./cloudflared-linux-amd64 tunnel --url http://localhost:8501
