"""
Intelligence Analysis Web App — pure Python, zero extra dependencies.

Usage:
    python -m src.app
    python -m src.app --port 8080 --model qwen2.5:7b

Opens a browser-based chat UI served at http://localhost:8080.
All inference stays local (Ollama). No cloud calls.
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import kuzu

from src.agent.analyst_agent import AnalystAgent
from src.storage import kuzu_store, vector_store

# ---------------------------------------------------------------------------
# HTML / JS chat UI (single-page, inline — no asset files needed)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligence Analysis Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }

  header { background: #1a1f2e; border-bottom: 1px solid #2d3748; padding: 14px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #63b3ed; }
  .badge { background: #2d3748; color: #90cdf4; font-size: 0.72rem; padding: 3px 8px; border-radius: 99px; }

  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }

  .msg { max-width: 820px; width: 100%; }
  .msg.user { align-self: flex-end; }
  .msg.assistant { align-self: flex-start; }

  .bubble { padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 0.92rem; white-space: pre-wrap; word-break: break-word; }
  .user .bubble   { background: #2b4c7e; color: #e2e8f0; border-bottom-right-radius: 3px; }
  .assistant .bubble { background: #1a1f2e; border: 1px solid #2d3748; border-bottom-left-radius: 3px; }

  .sources { margin-top: 8px; font-size: 0.75rem; color: #718096; }
  .sources span { background: #2d3748; padding: 2px 7px; border-radius: 99px; margin-right: 4px; display: inline-block; margin-top: 3px; }

  .thinking { color: #4a5568; font-style: italic; font-size: 0.85rem; }

  #input-row { display: flex; gap: 10px; padding: 16px 24px; background: #1a1f2e; border-top: 1px solid #2d3748; }
  #query { flex: 1; background: #2d3748; border: 1px solid #4a5568; border-radius: 8px; color: #e2e8f0; padding: 10px 14px; font-size: 0.92rem; resize: none; }
  #query:focus { outline: none; border-color: #63b3ed; }
  button { background: #2b4c7e; color: #e2e8f0; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-size: 0.92rem; white-space: nowrap; }
  button:hover { background: #3b5f9a; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }

  .examples { padding: 0 24px 16px; display: flex; flex-wrap: wrap; gap: 8px; }
  .examples button { background: #1a1f2e; border: 1px solid #2d3748; font-size: 0.8rem; padding: 6px 12px; color: #90cdf4; }
  .examples button:hover { border-color: #63b3ed; }
</style>
</head>
<body>

<header>
  <h1>&#128269; Intelligence Analysis Assistant</h1>
  <span class="badge">Qwen2.5-7b · Ollama</span>
  <span class="badge">KuzuDB · ChromaDB</span>
  <span class="badge">Air-gapped</span>
</header>

<div id="chat"></div>

<div class="examples">
  <button onclick="setQ('Who controls Northstar Trading Ltd and what transactions did they make?')">Northstar ownership</button>
  <button onclick="setQ('Trace the money from ACC-4471 to Bluewater Ventures Ltd.')">Money trail</button>
  <button onclick="setQ('What communications suggest coordination around the Rotterdam shipment?')">Rotterdam comms</button>
  <button onclick="setQ('Summarise all suspicious activity linked to Marcus Vane.')">Marcus Vane profile</button>
  <button onclick="setQ('What is the connection between Elena Ross and Shell Corp IO?')">Elena Ross links</button>
</div>

<div id="input-row">
  <textarea id="query" rows="2" placeholder="Ask an investigation question…"></textarea>
  <button id="send" onclick="send()">Send</button>
</div>

<script>
const chat = document.getElementById('chat');
const qEl  = document.getElementById('query');
const btn  = document.getElementById('send');

function setQ(text) { qEl.value = text; qEl.focus(); }

qEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

function addMsg(role, text, sources) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  if (sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.innerHTML = 'Sources: ' + [...new Set(sources)].sort().map(x => `<span>${x}</span>`).join('');
    div.appendChild(s);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function send() {
  const q = qEl.value.trim();
  if (!q) return;
  qEl.value = '';
  btn.disabled = true;

  addMsg('user', q);

  const thinkDiv = document.createElement('div');
  thinkDiv.className = 'msg assistant';
  thinkDiv.innerHTML = '<div class="bubble thinking">Investigating…</div>';
  chat.appendChild(thinkDiv);
  chat.scrollTop = chat.scrollHeight;

  try {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await resp.json();
    thinkDiv.remove();
    addMsg('assistant', data.answer, data.sources);
  } catch(e) {
    thinkDiv.remove();
    addMsg('assistant', 'Error: ' + e.message);
  }
  btn.disabled = false;
  chat.scrollTop = chat.scrollHeight;
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

_agent: AnalystAgent | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())
        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/query":
            self._send(404, "text/plain", b"Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send(400, "application/json", json.dumps({"error": "bad json"}).encode())
            return

        question = payload.get("question", "").strip()
        if not question:
            self._send(400, "application/json", json.dumps({"error": "empty question"}).encode())
            return

        try:
            resp = _agent.run(question)
            result = {
                "answer":  resp.answer,
                "sources": sorted(set(resp.sources)),
                "steps":   resp.steps,
                "tools":   [t["tool"] for t in resp.tool_calls_made],
            }
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as exc:
            self._send(500, "application/json", json.dumps({"error": str(exc)}).encode())

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligence Analysis Web App")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model",    default="qwen2.5:7b")
    args = parser.parse_args()

    global _agent

    print("Loading knowledge stores…", flush=True)
    try:
        kuzu_db   = kuzu_store.open_db()
        kuzu_conn = kuzu.Connection(kuzu_db)
        chroma    = vector_store.open_client()
    except Exception as exc:
        print(f"ERROR: Could not open stores: {exc}")
        print("Run 'python -m src.pipeline' first to populate the databases.")
        sys.exit(1)

    _agent = AnalystAgent(
        kuzu_conn=kuzu_conn,
        chroma_client=chroma,
        base_url=args.base_url,
        model=args.model,
    )

    server = HTTPServer((args.host, args.port), Handler)
    url    = f"http://{args.host}:{args.port}"

    print(f"Ready  →  {url}")
    print(f"Model  :  {args.model} @ {args.base_url}")
    print("Press Ctrl-C to stop.\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
