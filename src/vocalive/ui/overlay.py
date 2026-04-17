from __future__ import annotations

import asyncio
import contextlib
import html
import json
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
from string import Template
from urllib.parse import urlparse

from vocalive.config.settings import OverlaySettings
from vocalive.pipeline.events import ConversationEvent, ConversationEventSink


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_CHARACTER_IMAGE_PATH = _ASSETS_DIR / "character.png"
_CHARACTER_IMAGE_ROUTE = "/assets/character.png"


_CHARACTER_SVG = """
<svg viewBox="0 0 480 720" aria-hidden="true" role="img">
  <g class="tail">
    <path d="M100 520 C20 560, 26 668, 136 654 C194 646, 204 570, 160 544 C134 528, 136 498, 184 488" />
  </g>
  <g class="body">
    <path class="hoodie" d="M132 664 C116 584, 130 470, 208 422 C248 398, 322 398, 364 426 C438 476, 448 584, 424 664 Z" />
    <path class="hoodie-shadow" d="M192 432 C162 474, 150 528, 160 664 L232 664 C216 578, 218 474, 248 430 Z" />
    <path class="shirt" d="M212 470 C234 448, 284 448, 304 470 L292 586 C280 596, 236 596, 224 586 Z" />
    <path class="arm left" d="M150 516 C122 550, 112 612, 134 640 C152 664, 196 660, 210 632 C228 598, 220 534, 186 510 C174 502, 162 504, 150 516 Z" />
    <path class="arm right" d="M358 512 C392 538, 404 608, 382 642 C366 668, 322 664, 304 636 C284 602, 286 530, 320 510 C334 502, 346 502, 358 512 Z" />
    <ellipse class="paw" cx="188" cy="644" rx="30" ry="24" />
    <ellipse class="paw" cx="324" cy="644" rx="30" ry="24" />
  </g>
  <g class="head">
    <path class="ear" d="M172 176 C154 110, 182 74, 228 74 C250 76, 268 96, 278 126 C230 128, 196 146, 172 176 Z" />
    <path class="ear" d="M308 126 C318 96, 336 76, 358 74 C404 74, 432 110, 414 176 C390 146, 356 128, 308 126 Z" />
    <path class="inner-ear" d="M198 132 C208 104, 224 90, 246 90 C254 102, 260 116, 264 132 C236 132, 214 132, 198 132 Z" />
    <path class="inner-ear" d="M336 90 C358 90, 374 104, 384 132 C368 132, 346 132, 318 132 C322 116, 328 102, 336 90 Z" />
    <ellipse class="hair-back" cx="266" cy="294" rx="148" ry="164" />
    <ellipse class="face" cx="266" cy="270" rx="122" ry="128" />
    <path class="hair" d="M148 234 C154 156, 216 116, 300 118 C362 120, 424 162, 418 250 C374 210, 332 192, 282 194 C246 196, 228 244, 208 268 C198 242, 176 228, 148 234 Z" />
    <path class="bangs" d="M162 224 C192 180, 250 154, 334 164 C308 170, 286 190, 272 220 C252 192, 226 192, 198 224 C190 210, 176 206, 162 224 Z" />
    <circle class="headphone" cx="154" cy="244" r="54" />
    <circle class="headphone inner" cx="154" cy="244" r="34" />
    <circle class="headphone" cx="380" cy="244" r="54" />
    <circle class="headphone inner" cx="380" cy="244" r="34" />
    <path class="headband" d="M174 178 C210 128, 324 128, 360 178" />
    <path class="stripe" d="M252 150 L280 134 L290 164 L262 182 Z" />
    <path class="stripe" d="M218 190 L244 170 L252 206 L228 224 Z" />
    <path class="stripe" d="M292 194 L320 176 L328 208 L302 226 Z" />
    <ellipse class="blush" cx="204" cy="312" rx="22" ry="14" />
    <ellipse class="blush" cx="328" cy="312" rx="22" ry="14" />
    <ellipse class="eye white" cx="214" cy="272" rx="34" ry="44" />
    <ellipse class="eye white" cx="318" cy="272" rx="34" ry="44" />
    <ellipse class="eye iris" cx="214" cy="280" rx="22" ry="30" />
    <ellipse class="eye iris" cx="318" cy="280" rx="22" ry="30" />
    <circle class="eye pupil" cx="214" cy="288" r="12" />
    <circle class="eye pupil" cx="318" cy="288" r="12" />
    <circle class="eye shine" cx="206" cy="276" r="6" />
    <circle class="eye shine" cx="310" cy="276" r="6" />
    <rect class="nose" x="258" y="300" width="12" height="10" rx="5" />
    <path class="mouth" d="M246 334 C258 348, 274 350, 288 334" />
  </g>
  <g class="details">
    <path class="stripe body-stripe" d="M168 532 C190 520, 208 522, 228 534" />
    <path class="stripe body-stripe" d="M312 534 C334 520, 352 522, 372 536" />
    <path class="stripe body-stripe" d="M156 588 C178 576, 198 578, 220 590" />
    <path class="stripe body-stripe" d="M324 590 C344 578, 362 580, 382 592" />
  </g>
</svg>
""".strip()


_PAGE_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    :root {
      --caption-fill: #fff9fb;
      --caption-shadow: rgba(109, 58, 75, 0.96);
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      font-family: "Avenir Next", "Trebuchet MS", "Hiragino Maru Gothic ProN", "YuGothic", sans-serif;
      color: var(--ink);
      background: transparent;
      overflow: hidden;
    }

    body {
      position: relative;
      pointer-events: none;
    }

    .overlay {
      position: absolute;
      right: 0;
      bottom: 0;
      width: min(46vw, 560px);
      min-width: 280px;
      height: min(82vh, 860px);
      display: flex;
      align-items: end;
      justify-content: end;
      padding: 0 12px 12px 0;
    }

    .comment-wrap {
      position: absolute;
      right: 12px;
      bottom: min(9vh, 72px);
      width: min(calc(100% - 20px), 540px);
      display: flex;
      justify-content: center;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity 180ms ease, transform 180ms ease;
      z-index: 4;
    }

    body.show-comment .comment-wrap {
      opacity: 1;
      transform: translateY(0);
    }

    .assistant-text {
      min-height: 1em;
      width: 100%;
      font-family: "Arial Rounded MT Bold", "Marker Felt", "Hiragino Maru Gothic ProN", sans-serif;
      font-size: clamp(1.15rem, 2.5vw, 2.15rem);
      line-height: 1.24;
      letter-spacing: 0.02em;
      color: var(--caption-fill);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      text-align: center;
      text-shadow:
        0 3px 0 var(--caption-shadow),
        2px 0 0 var(--caption-shadow),
        -2px 0 0 var(--caption-shadow),
        0 -2px 0 var(--caption-shadow),
        0 8px 18px rgba(70, 42, 56, 0.34);
    }

    .character-shell {
      position: relative;
      width: 100%;
      height: 100%;
    }

    .character-frame {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: end;
      justify-content: end;
    }

    .character-card {
      position: relative;
      width: auto;
      height: clamp(140px, 58vmin, 840px);
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 2 / 3;
    }

    .character-figure {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 100%;
      height: 100%;
      transform-origin: center bottom;
      transition: transform 180ms ease;
    }

    body.speaking .character-figure {
      transform: translateY(-4px) rotate(-1deg);
      animation: bob 1.05s ease-in-out infinite;
    }

    .character-figure svg,
    .character-figure img {
      width: 100%;
      height: 100%;
      overflow: visible;
    }

    .character-figure img {
      object-fit: contain;
      object-position: center bottom;
      filter: drop-shadow(0 18px 30px rgba(115, 67, 43, 0.18));
      user-select: none;
      -webkit-user-drag: none;
    }

    body.speaking .mouth {
      animation: talk 180ms ease-in-out infinite alternate;
    }

    body.speaking .headphone {
      filter: drop-shadow(0 0 8px rgba(110, 215, 210, 0.36));
    }

    .tail path {
      fill: none;
      stroke: #c45f19;
      stroke-width: 26;
      stroke-linecap: round;
    }

    .hoodie {
      fill: #ef7c22;
    }

    .hoodie-shadow,
    .arm {
      fill: #c95f1a;
    }

    .shirt {
      fill: #f4f7ff;
    }

    .paw {
      fill: #ffd8b7;
    }

    .ear {
      fill: #9e4314;
    }

    .inner-ear {
      fill: #ffe2ca;
    }

    .hair-back,
    .hair,
    .bangs {
      fill: #ff8f1f;
    }

    .face {
      fill: #ffe5d2;
    }

    .headphone {
      fill: #5a88d9;
    }

    .headphone.inner {
      fill: #bcdaf8;
    }

    .headband {
      fill: none;
      stroke: #254a7a;
      stroke-width: 12;
      stroke-linecap: round;
    }

    .stripe {
      fill: #7b3214;
    }

    .body-stripe {
      fill: none;
      stroke: #8d3810;
      stroke-width: 10;
      stroke-linecap: round;
    }

    .blush {
      fill: rgba(255, 148, 148, 0.5);
    }

    .eye.white {
      fill: #fffefc;
    }

    .eye.iris {
      fill: #37a2e8;
    }

    .eye.pupil {
      fill: #11224f;
    }

    .eye.shine {
      fill: #ffffff;
    }

    .nose {
      fill: #dc8f68;
    }

    .mouth {
      fill: none;
      stroke: #7b3a30;
      stroke-width: 8;
      stroke-linecap: round;
      transform-origin: center;
    }

    @keyframes bob {
      0%, 100% { transform: translateY(-4px) rotate(-1deg); }
      50% { transform: translateY(-10px) rotate(1deg); }
    }

    @keyframes talk {
      from { transform: scaleX(0.92) scaleY(0.86); }
      to { transform: scaleX(1.08) scaleY(1.24); }
    }

    @media (max-width: 900px) {
      .overlay {
        width: min(58vw, 460px);
      }

      .comment-wrap {
        right: 8px;
        bottom: min(8vh, 56px);
        width: min(calc(100% - 12px), 430px);
      }
    }

    @media (max-width: 640px) {
      .overlay {
        width: min(72vw, 360px);
        min-width: 180px;
        height: min(70vh, 520px);
        padding: 0 4px 0 0;
      }

      .comment-wrap {
        right: 0;
        bottom: min(7vh, 38px);
        width: min(calc(100% - 4px), 320px);
      }

      .assistant-text {
        font-size: clamp(1rem, 4.8vw, 1.35rem);
      }
    }
  </style>
</head>
<body>
  <aside class="overlay" aria-label="$character_name overlay">
    <div class="comment-wrap">
      <div class="assistant-text" data-assistant-text></div>
    </div>
    <div class="character-shell" aria-label="Character panel">
      <div class="character-frame">
        <div class="character-card">
          <div class="character-figure" data-character>$character_markup</div>
        </div>
      </div>
    </div>
  </aside>

  <script>
    (function () {
      var assistantTextEl = document.querySelector("[data-assistant-text]");
      var root = document.body;
      var committedText = "";
      var animationToken = 0;

      function setStatus(status) {
        root.dataset.status = status || "idle";
        root.classList.toggle("speaking", status === "speaking");
        root.classList.toggle("show-comment", status === "speaking");
      }

      function stopAnimation() {
        animationToken += 1;
        root.classList.remove("speaking");
      }

      function resetAssistant(placeholder) {
        committedText = "";
        stopAnimation();
        assistantTextEl.textContent = placeholder || "";
        root.classList.remove("show-comment");
      }

      function animateChunk(chunkText, durationMs) {
        var chars = Array.from(chunkText || "");
        if (chars.length === 0) {
          return;
        }
        var baseText = committedText;
        var token = animationToken + 1;
        animationToken = token;
        setStatus("speaking");
        var startedAt = null;
        var totalDuration = Math.max(durationMs || 0, chars.length * 56, 340);

        function renderFrame(now) {
          if (token !== animationToken) {
            return;
          }
          if (startedAt === null) {
            startedAt = now;
          }
          var progress = Math.min(1, (now - startedAt) / totalDuration);
          var visibleCount = Math.max(1, Math.ceil(chars.length * progress));
          assistantTextEl.textContent = baseText + chars.slice(0, visibleCount).join("");
          root.classList.add("show-comment");
          if (progress < 1) {
            window.requestAnimationFrame(renderFrame);
            return;
          }
          committedText = baseText + chunkText;
          assistantTextEl.textContent = committedText;
        }

        window.requestAnimationFrame(renderFrame);
      }

      function applySnapshot(payload) {
        committedText = payload.assistant_text || "";
        assistantTextEl.textContent = payload.status === "speaking" ? committedText : "";
        root.classList.toggle(
          "show-comment",
          payload.status === "speaking" && Boolean(committedText.trim())
        );
        setStatus(payload.status || "idle");
      }

      function handleEvent(payload) {
        if (!payload || !payload.type) {
          return;
        }
        if (payload.type === "snapshot") {
          applySnapshot(payload);
          return;
        }
        if (payload.type === "transcription_ready") {
          resetAssistant("");
          return;
        }
        if (payload.type === "response_ready") {
          resetAssistant("");
          return;
        }
        if (payload.type === "assistant_chunk_started") {
          animateChunk(payload.text || "", payload.duration_ms);
          return;
        }
        if (payload.type === "assistant_message_committed") {
          stopAnimation();
          committedText = "";
          assistantTextEl.textContent = "";
          root.classList.remove("show-comment");
          setStatus("idle");
          return;
        }
        if (payload.type === "turn_interrupted" || payload.type === "turn_cancelled") {
          stopAnimation();
          assistantTextEl.textContent = "";
          committedText = "";
          root.classList.remove("show-comment");
          return;
        }
        if (payload.type === "session_idle") {
          stopAnimation();
          assistantTextEl.textContent = "";
          committedText = "";
          root.classList.remove("show-comment");
          setStatus("idle");
        }
      }

      var source = new EventSource("/events");
      source.onmessage = function (message) {
        handleEvent(JSON.parse(message.data));
      };
      source.onerror = function () {
        setStatus("interrupted");
      };
    }());
  </script>
</body>
</html>
"""
)


class OverlayServer(ConversationEventSink):
    def __init__(self, settings: OverlaySettings) -> None:
        self.settings = settings
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._subscriber_lock = threading.Lock()
        self._subscribers: set[Queue[bytes | None]] = set()
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict[str, object] = {
            "type": "snapshot",
            "session_id": None,
            "turn_id": None,
            "user_text": "",
            "assistant_text": "",
            "status": "idle",
        }
        self._active_assistant_turn_id: int | None = None

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Overlay server has not been started")
        host, port = self._httpd.server_address[:2]
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{display_host}:{port}/"

    async def start(self) -> None:
        if self._httpd is not None:
            return
        server = ThreadingHTTPServer((self.settings.host, self.settings.port), self._build_handler())
        server.daemon_threads = True
        self._httpd = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="vocalive-overlay",
            daemon=True,
        )
        self._thread.start()
        if self.settings.auto_open:
            await asyncio.to_thread(self._open_browser)

    async def stop(self) -> None:
        await asyncio.to_thread(self.close)

    def close(self) -> None:
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            _push_message(subscriber, None)
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def emit(self, event: ConversationEvent) -> None:
        payload = asdict(event)
        if not self._apply_to_snapshot(payload):
            return
        message = _format_sse(payload)
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            _push_message(subscriber, message)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        overlay = self

        class OverlayHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    overlay._serve_index(self)
                    return
                if path == _CHARACTER_IMAGE_ROUTE:
                    overlay._serve_character_image(self)
                    return
                if path == "/events":
                    overlay._serve_events(self)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return None

        return OverlayHandler

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        page = render_overlay_page(self.settings).encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(page)))
        handler.end_headers()
        handler.wfile.write(page)

    def _serve_character_image(self, handler: BaseHTTPRequestHandler) -> None:
        if not _CHARACTER_IMAGE_PATH.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND)
            return
        image_bytes = _CHARACTER_IMAGE_PATH.read_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/png")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(image_bytes)))
        handler.end_headers()
        handler.wfile.write(image_bytes)

    def _serve_events(self, handler: BaseHTTPRequestHandler) -> None:
        subscriber: Queue[bytes | None] = Queue(maxsize=64)
        with self._subscriber_lock:
            self._subscribers.add(subscriber)
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        try:
            handler.wfile.write(_format_sse(self._snapshot_payload()))
            handler.wfile.flush()
            while True:
                try:
                    message = subscriber.get(timeout=10.0)
                except Empty:
                    handler.wfile.write(b": keep-alive\n\n")
                    handler.wfile.flush()
                    continue
                if message is None:
                    return
                handler.wfile.write(message)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            with self._subscriber_lock:
                self._subscribers.discard(subscriber)

    def _snapshot_payload(self) -> dict[str, object]:
        with self._snapshot_lock:
            return dict(self._snapshot)

    def _apply_to_snapshot(self, payload: dict[str, object]) -> bool:
        with self._snapshot_lock:
            if payload.get("turn_id") is not None:
                event_turn_id = int(payload["turn_id"])
            else:
                event_turn_id = None
            event_type = payload.get("type")
            active_turn_id = self._active_assistant_turn_id
            if (
                active_turn_id is not None
                and event_turn_id is not None
                and event_turn_id != active_turn_id
                and event_type in {
                    "transcription_ready",
                    "response_ready",
                    "assistant_message_committed",
                    "turn_interrupted",
                    "turn_cancelled",
                }
            ):
                return False
            if payload.get("session_id") is not None:
                self._snapshot["session_id"] = payload["session_id"]
            if event_type == "transcription_ready":
                self._snapshot["turn_id"] = event_turn_id
                self._snapshot["user_text"] = payload.get("text") or ""
                self._snapshot["assistant_text"] = ""
                self._snapshot["status"] = "thinking"
                self._active_assistant_turn_id = None
            elif event_type == "response_ready":
                self._snapshot["turn_id"] = event_turn_id
                self._snapshot["assistant_text"] = ""
                self._snapshot["status"] = "thinking"
                self._active_assistant_turn_id = None
            elif event_type == "assistant_chunk_started":
                chunk_text = payload.get("text") or ""
                base_text = ""
                if active_turn_id == event_turn_id:
                    base_text = str(self._snapshot.get("assistant_text") or "")
                self._snapshot["turn_id"] = event_turn_id
                self._snapshot["assistant_text"] = f"{base_text}{chunk_text}"
                self._snapshot["status"] = "speaking"
                self._active_assistant_turn_id = event_turn_id
            elif event_type in {"turn_interrupted", "turn_cancelled"}:
                self._snapshot["turn_id"] = event_turn_id
                self._snapshot["assistant_text"] = ""
                self._snapshot["status"] = "interrupted"
                self._active_assistant_turn_id = None
            elif event_type == "assistant_message_committed":
                self._snapshot["turn_id"] = event_turn_id
                self._snapshot["assistant_text"] = payload.get("text") or ""
                self._snapshot["status"] = "idle"
                self._active_assistant_turn_id = None
            elif event_type == "session_idle":
                self._snapshot["status"] = "idle"
            return True

    def _open_browser(self) -> None:
        with contextlib.suppress(Exception):
            webbrowser.open(self.url, new=1)


def _format_sse(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def render_overlay_page(settings: OverlaySettings) -> str:
    return _PAGE_TEMPLATE.substitute(
        title=html.escape(settings.title),
        character_name=html.escape(settings.character_name),
        character_markup=_render_character_markup(),
    )


def character_image_path() -> Path:
    return _CHARACTER_IMAGE_PATH


def _render_character_markup() -> str:
    if _CHARACTER_IMAGE_PATH.is_file():
        return (
            f'<img src="{_CHARACTER_IMAGE_ROUTE}" alt="'
            "Overlay character"
            '" loading="eager" decoding="async" />'
        )
    return _CHARACTER_SVG


def _push_message(subscriber: Queue[bytes | None], message: bytes | None) -> None:
    with contextlib.suppress(Full):
        subscriber.put_nowait(message)
        return
    with contextlib.suppress(Empty):
        subscriber.get_nowait()
    with contextlib.suppress(Full):
        subscriber.put_nowait(message)
