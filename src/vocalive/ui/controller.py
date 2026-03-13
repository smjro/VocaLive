from __future__ import annotations

import asyncio
import contextlib
import html
import json
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from vocalive.audio.input import AudioInput
from vocalive.config.controller_store import ControllerConfigStore
from vocalive.config.settings import (
    AppSettings,
    InputProvider,
    controller_default_values,
    controller_setting_schema,
    normalize_controller_values,
)
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.runtime import build_audio_input, build_orchestrator, build_overlay
from vocalive.ui.overlay import OverlayServer
from vocalive.util.logging import configure_logging


_CONFIG_PATH_PLACEHOLDER = "__VOCALIVE_CONTROLLER_CONFIG_PATH__"


_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VocaLive Controller</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 250, 243, 0.92);
      --panel-strong: #fffdf8;
      --ink: #1f1a15;
      --muted: #65584a;
      --line: rgba(83, 64, 43, 0.14);
      --accent: #c75b12;
      --accent-deep: #8f3e00;
      --success: #21603d;
      --danger: #8f1d1d;
      --shadow: 0 16px 40px rgba(102, 69, 35, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      min-height: 100%;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 206, 154, 0.42), transparent 36%),
        radial-gradient(circle at top right, rgba(255, 247, 220, 0.78), transparent 28%),
        linear-gradient(180deg, #f9f5ed 0%, var(--bg) 100%);
      font-family: "Avenir Next", "Trebuchet MS", "Hiragino Sans", "Yu Gothic", sans-serif;
    }

    body {
      padding: 24px;
    }

    .shell {
      max-width: 1320px;
      margin: 0 auto;
      display: grid;
      gap: 20px;
    }

    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 24px 28px;
      display: grid;
      gap: 16px;
    }

    .hero-top {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }

    .eyebrow {
      margin: 0;
      font-size: 0.8rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent-deep);
    }

    h1 {
      margin: 4px 0 0;
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .subcopy {
      margin: 0;
      max-width: 58rem;
      color: var(--muted);
      line-height: 1.55;
    }

    .status-card {
      min-width: 280px;
      padding: 16px 18px;
      background: var(--panel-strong);
      border-radius: 18px;
      border: 1px solid var(--line);
    }

    .status-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }

    .status-pill {
      padding: 7px 12px;
      border-radius: 999px;
      font-size: 0.83rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: white;
      background: var(--accent);
    }

    .status-pill[data-status="running"] {
      background: var(--success);
    }

    .status-pill[data-status="error"] {
      background: var(--danger);
    }

    .status-pill[data-status="stopping"],
    .status-pill[data-status="starting"] {
      background: var(--accent-deep);
    }

    .status-meta {
      margin: 0;
      font-size: 0.92rem;
      color: var(--muted);
      line-height: 1.5;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }

    button {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, box-shadow 120ms ease;
    }

    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(129, 73, 27, 0.16);
    }

    button:disabled {
      opacity: 0.45;
      cursor: default;
      box-shadow: none;
    }

    .button-primary {
      background: linear-gradient(135deg, #dc6f1e, #c44f00);
      color: white;
    }

    .button-secondary {
      background: #f2e7d6;
      color: var(--ink);
      border: 1px solid rgba(116, 88, 58, 0.18);
    }

    .button-danger {
      background: #4d3a2f;
      color: white;
    }

    .alert {
      display: none;
      padding: 14px 16px;
      border-radius: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
    }

    .alert.show {
      display: block;
    }

    .alert-error {
      background: rgba(143, 29, 29, 0.1);
      border: 1px solid rgba(143, 29, 29, 0.18);
      color: var(--danger);
    }

    .alert-info {
      background: rgba(199, 91, 18, 0.08);
      border: 1px solid rgba(199, 91, 18, 0.16);
      color: var(--accent-deep);
    }

    .panel {
      padding: 22px;
    }

    .panel-head {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 1.2rem;
      letter-spacing: -0.03em;
    }

    .panel-head p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .groups {
      display: grid;
      gap: 18px;
    }

    .group {
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(108, 82, 53, 0.12);
      border-radius: 20px;
      padding: 18px;
    }

    .group h3 {
      margin: 0 0 14px;
      font-size: 1rem;
      text-transform: capitalize;
      letter-spacing: 0.02em;
    }

    .field-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }

    .field {
      display: grid;
      gap: 8px;
      min-width: 0;
    }

    .field.wide {
      grid-column: 1 / -1;
    }

    .field label {
      display: grid;
      gap: 4px;
      min-width: 0;
    }

    .field-meta {
      display: grid;
      gap: 6px;
      min-width: 0;
    }

    .field-name {
      font-size: 0.86rem;
      font-weight: 700;
      letter-spacing: 0.01em;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .field-default {
      font-size: 0.78rem;
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .field-control {
      display: block;
      min-width: 0;
    }

    .field-help {
      min-width: 0;
      padding: 10px 12px;
      background: rgba(255, 248, 239, 0.92);
      border: 1px solid rgba(199, 91, 18, 0.14);
      border-radius: 14px;
    }

    .field-help summary {
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--accent-deep);
      list-style: none;
    }

    .field-help summary::-webkit-details-marker {
      display: none;
    }

    .field-help[open] summary {
      margin-bottom: 8px;
    }

    .field-help p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .field-help p + p {
      margin-top: 6px;
    }

    input[type="text"],
    input[type="password"],
    input[type="number"],
    select,
    textarea {
      width: 100%;
      min-width: 0;
      border: 1px solid rgba(104, 79, 50, 0.18);
      border-radius: 14px;
      padding: 11px 12px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.92);
    }

    textarea {
      min-height: 124px;
      resize: vertical;
      line-height: 1.45;
    }

    .bool-field {
      min-height: 100%;
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 10px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(104, 79, 50, 0.18);
      border-radius: 14px;
    }

    .bool-field input {
      width: 18px;
      height: 18px;
      margin: 0;
    }

    .bool-copy {
      display: grid;
      gap: 4px;
      min-width: 0;
      color: var(--muted);
      overflow-wrap: anywhere;
      word-break: break-word;
    }

    .config-path {
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      word-break: break-all;
    }

    @media (max-width: 720px) {
      body {
        padding: 16px;
      }

      .hero, .panel {
        border-radius: 20px;
      }

      .hero {
        padding: 20px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-top">
        <div>
          <p class="eyebrow">Local Runtime Controller</p>
          <h1>VocaLive</h1>
          <p class="subcopy">
            設定を保存し、必要なときだけ会話ランタイムを開始・停止します。
            `stdin` シェルは GUI からは起動せず、`python -m vocalive run` を使います。
          </p>
        </div>
        <div class="status-card">
          <div class="status-row">
            <strong>Runtime</strong>
            <span id="status-pill" class="status-pill" data-status="stopped">stopped</span>
          </div>
          <p id="status-meta" class="status-meta">保存済み設定を読み込み待ちです。</p>
        </div>
      </div>
      <div class="actions">
        <button id="save-button" class="button-secondary" type="button">Save Settings</button>
        <button id="start-button" class="button-primary" type="button">Start Conversation</button>
        <button id="stop-button" class="button-danger" type="button">Stop Conversation</button>
      </div>
      <div id="warning-banner" class="alert alert-info"></div>
      <div id="error-banner" class="alert alert-error"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div>
          <h2>Controller Config</h2>
          <p>保存先: <span id="config-path" class="config-path">__VOCALIVE_CONTROLLER_CONFIG_PATH__</span></p>
        </div>
        <p>フォームは全設定を env 名ベースで編集します。</p>
      </div>
      <div id="groups" class="groups"></div>
    </section>
  </div>

  <script>
    const state = {
      schema: [],
      values: {},
      runtime: null,
      busy: false,
    };

    const GROUP_LABELS = {
      general: "General",
      providers: "Providers",
      queue: "Queue",
      conversation: "Conversation",
      context: "Context",
      input: "Input",
      application_audio: "Application Audio",
      output: "Output",
      overlay: "Overlay",
      reply: "Reply Policy",
      gemini: "Gemini",
      screen_capture: "Screen Capture",
      moonshine: "Moonshine",
      aivis: "Aivis",
    };

    const groupsRoot = document.getElementById("groups");
    const statusPill = document.getElementById("status-pill");
    const statusMeta = document.getElementById("status-meta");
    const warningBanner = document.getElementById("warning-banner");
    const errorBanner = document.getElementById("error-banner");
    const saveButton = document.getElementById("save-button");
    const startButton = document.getElementById("start-button");
    const stopButton = document.getElementById("stop-button");

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function renderInlineMarkup(value) {
      return escapeHtml(value).replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    function formatDefault(field) {
      const label = field.default_label === null || field.default_label === undefined
        ? (field.default_raw === null ? "unset" : field.default_raw)
        : field.default_label;
      return "default: " + label;
    }

    function renderFieldInfo(field) {
      const lines = [];
      if (field.description) {
        lines.push('<p>' + renderInlineMarkup(field.description) + '</p>');
      }
      if (field.kind === "enum" && Array.isArray(field.options) && field.options.length > 0) {
        const options = field.options.map(function(option) {
          return "<code>" + escapeHtml(option) + "</code>";
        }).join(", ");
        lines.push("<p>Allowed: " + options + "</p>");
      } else if (field.kind === "bool") {
        lines.push("<p>Allowed: <code>true</code>, <code>false</code></p>");
      }
      if (field.nullable) {
        lines.push("<p>Empty value: allowed</p>");
      }
      if (field.kind === "tuple") {
        lines.push("<p>Format: comma-separated values</p>");
      }
      if (lines.length === 0) {
        return "";
      }
      return '<details class="field-help"><summary>Info</summary>' + lines.join("") + "</details>";
    }

    function renderField(field) {
      const fieldId = "field-" + field.env_name;
      const currentValue = state.values[field.env_name];
      const resolvedValue = currentValue === null || currentValue === undefined
        ? (field.default_raw ?? "")
        : String(currentValue);
      const wide = field.multiline || field.kind === "tuple";
      const defaultHtml = '<span class="field-default">' + escapeHtml(formatDefault(field)) + '</span>';
      const infoHtml = renderFieldInfo(field);
      const metaHtml = `
        <div class="field-meta">
          <span class="field-name">${escapeHtml(field.env_name)}</span>
          ${defaultHtml}
          ${infoHtml}
        </div>
      `;
      const commonAttrs = 'id="' + escapeHtml(fieldId) + '" data-env-name="' + escapeHtml(field.env_name) + '"';

      if (field.kind === "bool") {
        const checked = resolvedValue.toLowerCase() === "true" ? " checked" : "";
        return `
          <div class="field">
            ${metaHtml}
            <label class="bool-field" for="${escapeHtml(fieldId)}">
                <input type="checkbox" ${commonAttrs}${checked} />
                <span class="bool-copy"><code>true / false</code></span>
            </label>
          </div>
        `;
      }

      let controlHtml = "";
      if (field.kind === "enum") {
        const options = field.options.map(function(option) {
          const selected = option === resolvedValue ? ' selected' : '';
          return '<option value="' + escapeHtml(option) + '"' + selected + '>' + escapeHtml(option) + '</option>';
        }).join("");
        controlHtml = '<select ' + commonAttrs + '>' + options + '</select>';
      } else if (field.multiline || field.kind === "tuple") {
        controlHtml = '<textarea ' + commonAttrs + '>' + escapeHtml(resolvedValue) + '</textarea>';
      } else if (field.kind === "int" || field.kind === "float") {
        const inputMode = field.kind === "int" ? "numeric" : "decimal";
        controlHtml = '<input type="number" inputmode="' + inputMode + '" step="any" ' + commonAttrs + ' value="' + escapeHtml(resolvedValue) + '" />';
      } else {
        const type = field.secret ? "password" : "text";
        controlHtml = '<input type="' + type + '" ' + commonAttrs + ' value="' + escapeHtml(resolvedValue) + '" />';
      }

      return `
        <div class="field ${wide ? "wide" : ""}">
          ${metaHtml}
          <label class="field-control" for="${escapeHtml(fieldId)}">
            ${controlHtml}
          </label>
        </div>
      `;
    }

    function renderGroups() {
      const grouped = new Map();
      state.schema.forEach(function(field) {
        if (!grouped.has(field.group)) {
          grouped.set(field.group, []);
        }
        grouped.get(field.group).push(field);
      });
      groupsRoot.innerHTML = Array.from(grouped.entries()).map(function(entry) {
        const groupName = entry[0];
        const fields = entry[1];
        return `
          <section class="group">
            <h3>${escapeHtml(GROUP_LABELS[groupName] || groupName)}</h3>
            <div class="field-grid">
              ${fields.map(renderField).join("")}
            </div>
          </section>
        `;
      }).join("");
    }

    function collectValues() {
      const values = {};
      state.schema.forEach(function(field) {
        const control = groupsRoot.querySelector('[data-env-name="' + field.env_name + '"]');
        if (!control) {
          return;
        }
        if (field.kind === "bool") {
          values[field.env_name] = control.checked ? "true" : "false";
          return;
        }
        values[field.env_name] = control.value;
      });
      return values;
    }

    function setBusy(busy) {
      state.busy = busy;
      saveButton.disabled = busy;
      startButton.disabled = busy;
      stopButton.disabled = busy;
    }

    function setError(message) {
      if (message) {
        errorBanner.textContent = message;
        errorBanner.classList.add("show");
      } else {
        errorBanner.textContent = "";
        errorBanner.classList.remove("show");
      }
    }

    function setWarning(message) {
      if (message) {
        warningBanner.textContent = message;
        warningBanner.classList.add("show");
      } else {
        warningBanner.textContent = "";
        warningBanner.classList.remove("show");
      }
    }

    function updateRuntime(snapshot) {
      state.runtime = snapshot;
      const status = snapshot && snapshot.status ? snapshot.status : "stopped";
      statusPill.textContent = status;
      statusPill.dataset.status = status;
      let lines = [];
      if (snapshot && snapshot.input_label) {
        lines.push("Input: " + snapshot.input_label);
      }
      if (snapshot && snapshot.overlay_url) {
        lines.push("Overlay: " + snapshot.overlay_url);
      }
      if (snapshot && snapshot.error) {
        lines.push("Last error: " + snapshot.error);
      }
      if (lines.length === 0) {
        lines.push("保存済み設定を読み込み待ちです。");
      }
      statusMeta.textContent = lines.join(" | ");
    }

    async function requestJson(url, options) {
      const response = await fetch(url, Object.assign({
        headers: {
          "Content-Type": "application/json",
        },
      }, options || {}));
      const payload = await response.json().catch(function() {
        return {};
      });
      if (!response.ok) {
        const message = payload && payload.error ? payload.error : response.statusText;
        throw new Error(message);
      }
      return payload;
    }

    async function loadInitialData() {
      const [schemaPayload, configPayload, runtimePayload] = await Promise.all([
        requestJson("/api/config/schema"),
        requestJson("/api/config"),
        requestJson("/api/runtime/state"),
      ]);
      state.schema = schemaPayload.schema || [];
      state.values = configPayload.values || {};
      renderGroups();
      updateRuntime(runtimePayload.runtime || runtimePayload);
      setWarning(configPayload.warning || "");
    }

    async function refreshRuntime() {
      try {
        const payload = await requestJson("/api/runtime/state");
        updateRuntime(payload.runtime || payload);
      } catch (error) {
        setError(String(error.message || error));
      }
    }

    async function saveValues() {
      setBusy(true);
      setError("");
      try {
        const payload = await requestJson("/api/config", {
          method: "PUT",
          body: JSON.stringify({ values: collectValues() }),
        });
        state.values = payload.values || collectValues();
        renderGroups();
        setWarning(payload.warning || "");
      } finally {
        setBusy(false);
      }
    }

    async function startRuntime() {
      setBusy(true);
      setError("");
      try {
        const payload = await requestJson("/api/runtime/start", {
          method: "POST",
          body: JSON.stringify({ values: collectValues() }),
        });
        state.values = payload.values || collectValues();
        renderGroups();
        updateRuntime(payload.runtime);
        setWarning(payload.warning || "");
      } finally {
        setBusy(false);
      }
    }

    async function stopRuntime() {
      setBusy(true);
      setError("");
      try {
        const payload = await requestJson("/api/runtime/stop", {
          method: "POST",
          body: JSON.stringify({}),
        });
        updateRuntime(payload.runtime);
      } finally {
        setBusy(false);
      }
    }

    saveButton.addEventListener("click", function() {
      saveValues().catch(function(error) {
        setError(String(error.message || error));
      });
    });

    startButton.addEventListener("click", function() {
      startRuntime().catch(function(error) {
        setError(String(error.message || error));
      });
    });

    stopButton.addEventListener("click", function() {
      stopRuntime().catch(function(error) {
        setError(String(error.message || error));
      });
    });

    loadInitialData().catch(function(error) {
      setError(String(error.message || error));
      state.values = {};
      state.schema = [];
    });
    window.setInterval(refreshRuntime, 1500);
  </script>
</body>
</html>
"""


@dataclass
class _ActiveRuntime:
    settings: AppSettings
    orchestrator: ConversationOrchestrator
    audio_input: AudioInput
    overlay: OverlayServer | None
    reader_task: asyncio.Task[None]
    input_label: str | None


class ControllerRuntimeManager:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._state: dict[str, str | None] = {
            "status": "stopped",
            "error": None,
            "input_label": None,
            "overlay_url": None,
        }
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._started = False
        self._active_runtime: _ActiveRuntime | None = None
        self._lifecycle_lock: asyncio.Lock | None = None

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vocalive-controller-runtime",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        future.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        thread = self._thread
        if thread is not None and thread.is_alive():
          thread.join(timeout=1.0)
        self._thread = None
        self._started = False

    def snapshot(self) -> dict[str, str | None]:
        with self._state_lock:
            return dict(self._state)

    def start_runtime(self, values: dict[str, str | None]) -> dict[str, str | None]:
        self.start()
        future = asyncio.run_coroutine_threadsafe(
            self._start_runtime(values),
            self._loop,
        )
        return future.result()

    def stop_runtime(self) -> dict[str, str | None]:
        if not self._started:
            return self.snapshot()
        future = asyncio.run_coroutine_threadsafe(self._stop_runtime(), self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._lifecycle_lock = asyncio.Lock()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        self._loop.close()

    async def _shutdown(self) -> None:
        await self._stop_runtime()

    async def _start_runtime(self, values: dict[str, str | None]) -> dict[str, str | None]:
        if self._lifecycle_lock is None:
            raise RuntimeError("Controller runtime loop has not been initialized")
        async with self._lifecycle_lock:
            if self._active_runtime is not None:
                raise RuntimeError("Conversation runtime is already running")
            normalized = normalize_controller_values(values)
            settings = AppSettings.from_mapping(normalized)
            if settings.input.provider is InputProvider.STDIN:
                raise ValueError(
                    "Controller mode does not support VOCALIVE_INPUT_PROVIDER=stdin. "
                    "Use `python -m vocalive run` for the stdin shell."
                )
            configure_logging(settings.log_level)
            self._set_state(status="starting", error=None, input_label=None, overlay_url=None)

            overlay: OverlayServer | None = None
            orchestrator: ConversationOrchestrator | None = None
            audio_input: AudioInput | None = None
            try:
                overlay = build_overlay(settings)
                orchestrator = build_orchestrator(settings, event_sink=overlay)
                audio_input = build_audio_input(settings)
                if audio_input is None:
                    raise ValueError(
                        "Controller mode requires a live input source. "
                        "Use `python -m vocalive run` for stdin mode."
                    )
                if overlay is not None:
                    await overlay.start()
                await orchestrator.start()
                audio_input.set_speech_start_handler(orchestrator.handle_user_speech_start)
                input_label = await audio_input.start()
                reader_task = asyncio.create_task(
                    self._reader_loop(audio_input, orchestrator),
                    name="vocalive-controller-live-input",
                )
                self._active_runtime = _ActiveRuntime(
                    settings=settings,
                    orchestrator=orchestrator,
                    audio_input=audio_input,
                    overlay=overlay,
                    reader_task=reader_task,
                    input_label=input_label,
                )
                self._set_state(
                    status="running",
                    error=None,
                    input_label=input_label,
                    overlay_url=overlay.url if overlay is not None else None,
                )
                return self.snapshot()
            except Exception as exc:
                await self._cleanup_partial_runtime(
                    orchestrator=orchestrator,
                    audio_input=audio_input,
                    overlay=overlay,
                )
                self._set_state(
                    status="error",
                    error=str(exc),
                    input_label=None,
                    overlay_url=None,
                )
                raise

    async def _stop_runtime(self) -> dict[str, str | None]:
        if self._lifecycle_lock is None:
            return self.snapshot()
        async with self._lifecycle_lock:
            runtime = self._active_runtime
            if runtime is None:
                self._set_state(
                    status="stopped",
                    error=None,
                    input_label=None,
                    overlay_url=None,
                )
                return self.snapshot()
            self._set_state(
                status="stopping",
                error=None,
                input_label=runtime.input_label,
                overlay_url=runtime.overlay.url if runtime.overlay is not None else None,
            )
            await self._stop_runtime_components(runtime, await_reader=True)
            self._active_runtime = None
            self._set_state(
                status="stopped",
                error=None,
                input_label=None,
                overlay_url=None,
            )
            return self.snapshot()

    async def _reader_loop(
        self,
        audio_input: AudioInput,
        orchestrator: ConversationOrchestrator,
    ) -> None:
        error_message: str | None = None
        try:
            while True:
                segment = await audio_input.read()
                if segment is None:
                    break
                await orchestrator.submit_utterance(segment)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            error_message = str(exc)
        if self.snapshot().get("status") == "stopping":
            return
        await self._handle_reader_exit(error_message)

    async def _handle_reader_exit(self, error_message: str | None) -> None:
        if self._lifecycle_lock is None:
            return
        async with self._lifecycle_lock:
            runtime = self._active_runtime
            if runtime is None:
                return
            if self.snapshot().get("status") == "stopping":
                return
            self._set_state(
                status="stopping",
                error=error_message,
                input_label=runtime.input_label,
                overlay_url=runtime.overlay.url if runtime.overlay is not None else None,
            )
            await self._stop_runtime_components(runtime, await_reader=False)
            self._active_runtime = None
            self._set_state(
                status="error" if error_message else "stopped",
                error=error_message,
                input_label=None,
                overlay_url=None,
            )

    async def _cleanup_partial_runtime(
        self,
        *,
        orchestrator: ConversationOrchestrator | None,
        audio_input: AudioInput | None,
        overlay: OverlayServer | None,
    ) -> None:
        if audio_input is not None:
            with contextlib.suppress(Exception):
                await audio_input.close()
        if orchestrator is not None:
            with contextlib.suppress(Exception):
                await orchestrator.stop()
        if overlay is not None:
            with contextlib.suppress(Exception):
                await overlay.stop()

    async def _stop_runtime_components(
        self,
        runtime: _ActiveRuntime,
        *,
        await_reader: bool,
    ) -> None:
        with contextlib.suppress(Exception):
            await runtime.audio_input.close()
        if await_reader and runtime.reader_task is not asyncio.current_task():
            with contextlib.suppress(Exception):
                await runtime.reader_task
        with contextlib.suppress(Exception):
            await runtime.orchestrator.stop()
        if runtime.overlay is not None:
            with contextlib.suppress(Exception):
                await runtime.overlay.stop()

    def _set_state(
        self,
        *,
        status: str,
        error: str | None,
        input_label: str | None,
        overlay_url: str | None,
    ) -> None:
        with self._state_lock:
            self._state = {
                "status": status,
                "error": error,
                "input_label": input_label,
                "overlay_url": overlay_url,
            }


class ControllerServer:
    def __init__(
        self,
        store: ControllerConfigStore | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        auto_open: bool = True,
    ) -> None:
        self.store = store or ControllerConfigStore()
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self.runtime_manager = ControllerRuntimeManager()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("Controller server has not been started")
        host, port = self._httpd.server_address[:2]
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{display_host}:{port}/"

    async def start(self) -> None:
        if self._httpd is not None:
            return
        self.runtime_manager.start()
        server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        server.daemon_threads = True
        self._httpd = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="vocalive-controller-http",
            daemon=True,
        )
        self._thread.start()
        if self.auto_open:
            await asyncio.to_thread(self._open_browser)

    async def stop(self) -> None:
        await asyncio.to_thread(self.close)

    def close(self) -> None:
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        self.runtime_manager.close()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        controller = self

        class ControllerHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    controller._serve_index(self)
                    return
                if path == "/api/config/schema":
                    controller._serve_json(
                        self,
                        {"schema": controller_setting_schema()},
                    )
                    return
                if path == "/api/config":
                    values, warning = controller._load_values_with_warning()
                    controller._serve_json(
                        self,
                        {
                            "values": values,
                            "warning": warning,
                            "config_path": str(controller.store.path),
                        },
                    )
                    return
                if path == "/api/runtime/state":
                    controller._serve_json(
                        self,
                        {"runtime": controller.runtime_manager.snapshot()},
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_PUT(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/api/config":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    payload = controller._read_json_payload(self)
                    values = controller._extract_values(payload)
                    normalized = controller._validate_values(values)
                    saved_values = controller.store.save_values(normalized)
                    controller._serve_json(
                        self,
                        {
                            "values": saved_values,
                            "warning": None,
                            "config_path": str(controller.store.path),
                        },
                    )
                except Exception as exc:
                    controller._serve_json_error(self, HTTPStatus.BAD_REQUEST, str(exc))

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                try:
                    if path == "/api/runtime/start":
                        payload = controller._read_json_payload(self)
                        values = controller._extract_values(payload)
                        normalized = controller._validate_values(values)
                        saved_values = controller.store.save_values(normalized)
                        runtime = controller.runtime_manager.start_runtime(saved_values)
                        controller._serve_json(
                            self,
                            {
                                "values": saved_values,
                                "runtime": runtime,
                                "warning": None,
                            },
                        )
                        return
                    if path == "/api/runtime/stop":
                        runtime = controller.runtime_manager.stop_runtime()
                        controller._serve_json(self, {"runtime": runtime})
                        return
                    self.send_error(HTTPStatus.NOT_FOUND)
                except Exception as exc:
                    controller._serve_json_error(self, HTTPStatus.BAD_REQUEST, str(exc))

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return None

        return ControllerHandler

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        page = _PAGE_TEMPLATE.replace(
            _CONFIG_PATH_PLACEHOLDER,
            html.escape(str(self.store.path)),
        ).encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(page)))
        handler.end_headers()
        handler.wfile.write(page)

    def _serve_json(self, handler: BaseHTTPRequestHandler, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _serve_json_error(
        self,
        handler: BaseHTTPRequestHandler,
        status: HTTPStatus,
        message: str,
    ) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _read_json_payload(self, handler: BaseHTTPRequestHandler) -> dict[str, object]:
        content_length = int(handler.headers.get("Content-Length", "0") or "0")
        if content_length < 1:
            return {}
        raw_body = handler.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _extract_values(self, payload: dict[str, object]) -> dict[str, str | None]:
        raw_values = payload.get("values", controller_default_values())
        if not isinstance(raw_values, dict):
            raise ValueError("Request body must include a `values` object")
        return {
            str(env_name): None if value is None else str(value)
            for env_name, value in raw_values.items()
        }

    def _validate_values(self, values: dict[str, str | None]) -> dict[str, str | None]:
        normalized = normalize_controller_values(values)
        AppSettings.from_mapping(normalized)
        return normalized

    def _load_values_with_warning(self) -> tuple[dict[str, str | None], str | None]:
        try:
            return self.store.load_values(), None
        except Exception as exc:
            return controller_default_values(), str(exc)

    def _open_browser(self) -> None:
        with contextlib.suppress(Exception):
            webbrowser.open(self.url, new=1)


async def run_controller() -> int:
    server = ControllerServer()
    await server.start()
    print(f"VocaLive controller: {server.url}")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await server.stop()
