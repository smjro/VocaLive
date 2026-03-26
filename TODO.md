# Refactor TODO

## 現在地

`src/vocalive/pipeline/orchestrator.py` はまだ大きいが、以下はすでに別モジュールへ切り出してある。

- `src/vocalive/pipeline/request_building.py`
  - 会話メッセージ構築
  - proactive 用メッセージ構築
  - screen capture 文面生成
  - session へ入れる文面整形
- `src/vocalive/pipeline/playback.py`
  - レスポンス正規化
  - 文分割
  - TTS 先読み再生
  - chunk event 発火
  - `tts` / `playback` メトリクス記録
- `src/vocalive/pipeline/interrupt_probe.py`
  - explicit interrupt probe
  - cached transcript hint の再利用
  - 背景 probe task の管理
- `src/vocalive/pipeline/submission.py`
  - debounce buffer
  - segment merge
  - segment duration 計算

つまり今の `orchestrator.py` は、以前よりは「司令塔」に近づいているが、まだ次の責務がまとまって残っている。

- proactive monitor / observation state
- turn 実行本体
- work-item の選別
- screen capture 実行
- application-audio 特有の submit policy

この TODO は「次にどこを切るか」と「何を壊しやすいか」を後で思い出せるように書いている。

2026-03-26 時点の確認:

- `pytest -q` -> `217 passed`
- `python3 -m compileall src tests` -> 成功

2026-03-26 更新:

- 1. proactive 周りを `src/vocalive/pipeline/proactive.py` へ切り出し
- 2. turn 実行本体を `src/vocalive/pipeline/turn_execution.py` へ切り出し
- 3. screen capture 実行パスを `src/vocalive/pipeline/screen_capture_turn.py` へ切り出し
- `src/vocalive/pipeline/orchestrator.py` は各モジュールへの委譲を主に担当する形へ整理
- 現在の再確認:
  - `pytest -q` -> `216 passed`
  - `python3 -m compileall src tests` -> 成功

## 次に触る優先順

### 1. proactive 周りを `pipeline/proactive.py` へ寄せる

今 `orchestrator.py` に残っている proactive 関連:

- `_record_proactive_microphone_observation`
- `_record_proactive_application_audio_observation`
- `_record_proactive_observation`
- `_consume_proactive_observation`
- `_clear_proactive_observations`
- `_discard_pending_proactive_turns`
- `_is_current_proactive_observation`
- `_is_assistant_idle_for_proactive_work`
- `_can_start_proactive_turn`
- `_should_run_proactive_request`
- `_run_proactive_monitor`
- `_maybe_enqueue_proactive_turn`
- `_poll_proactive_screen_if_due`
- `_proactive_screen_capture_supported`

ここを先に勧める理由:

- 条件分岐が多い割に、turn 本体とは責務がかなり違う
- 画面監視、idle 判定、cooldown、observation version 管理が 1 つの塊になっている
- `orchestrator` を読むときに proactive が最もノイズになっている

切り出し方のイメージ:

- `ProactiveCoordinator` か `ProactiveManager` を作る
- state も manager 側へ寄せる
- `orchestrator` からは
  - `record_microphone_observation(...)`
  - `record_application_observation(...)`
  - `maybe_enqueue(...)`
  - `poll_screen_if_due(...)`
  のような入口だけ呼ぶ形にする

壊しやすいポイント:

- user activity 後すぐ proactive が出ないこと
- proactive cooldown を守ること
- 画面差分がないときに同じ screenshot を繰り返さないこと
- active playback 中に proactive が割り込まないこと

見るテスト:

- `tests/unit/test_orchestrator.py` の proactive 関連

### 2. turn 実行本体を `pipeline/turn_execution.py` のような形へ寄せる

今まだ `orchestrator.py` に残っている大きい塊:

- `_process_turn`
- `_process_proactive_turn`

ここを後回しにしている理由:

- session commit のタイミングを壊すと挙動差が出やすい
- interruption / cancellation / playback 完了後 commit の関係が重要

ただし、最終的にはここも切った方がよい。

分けるときのイメージ:

- `TurnExecutor`
  - STT
  - user/application message commit
  - reply policy
  - current user parts 決定
  - request 構築
  - LLM
  - playback
  - assistant commit
- proactive は `execute_proactive_turn(...)` を別メソッドにする

絶対に維持したいこと:

- user message は STT 後に commit
- assistant message は playback 完了後に commit
- interrupted assistant reply は commit しない
- application audio の `context_only` は LLM/TTS を起動しない

見るテスト:

- `tests/unit/test_orchestrator.py` の session / interruption / context-only 系

### 3. screen capture 実行パスを `pipeline/screen_capture_turn.py` へ寄せる

純粋 helper は `request_building.py` に寄せたが、実行本体はまだ `orchestrator.py` に残っている。

対象:

- `_maybe_capture_current_user_parts`

ここを分ける理由:

- いまは logging、timing、cooldown、fingerprint、provider 呼び出しが混ざっている
- request-building と screen-capture execution は別責務

分けるときのイメージ:

- `CurrentTurnScreenCapture` か `ScreenCaptureCoordinator`
- 引数は
  - screen settings
  - current stage setter
  - metrics
  - logger
  - engine
  - last fingerprint / timestamp
- 戻り値は
  - current user parts
  - 更新後の fingerprint / timestamp

壊しやすいポイント:

- passive cooldown
- unchanged screenshot skip
- cancellation は失敗ログ扱いにしないこと

見るテスト:

- `tests/unit/test_orchestrator.py` の screen capture 関連

### 4. work loop と idle 制御を整理する

今の塊:

- `_run`
- `_await_next_work_item`
- `_set_idle_if_drained`

これは大きな問題ではないが、最終的に `orchestrator` を「部品を呼ぶだけ」にしたいなら整理したい。

意図:

- ingress queue
- application context queue
- proactive queue
- idle event
- cancellation token lifecycle

が 1 か所に集中している

ただし優先度は上 3 つより低い。ここは後でもよい。

### 5. application-audio 特有の submit policy をまとめる

今の塊:

- `_should_capture_application_audio_as_context`
- `_should_debounce_application_segment`
- `_should_skip_application_audio_segment`
- `_begin_application_audio_submission`
- `_submit_application_audio_segment`
- `_submit_application_audio_segment_now`
- `submit_application_context`

分ける理由:

- microphone と application-audio で policy が違う
- `context_only` と `respond` のルールが独立した関心事になっている

ここは `pipeline/application_audio_submission.py` のようにまとめると理解しやすい。

維持したいこと:

- `context_only` は session には積むが即応答しない
- `respond` は live turn として流す
- cooldown / min duration skip / debounce の意味を変えない

見るテスト:

- `tests/unit/test_orchestrator.py` の application-audio 関連

### 6. 小さな整理

後回しでよいが、やると読みやすくなるもの:

- `_assistant_names_for_interrupt` を `interrupt_probe.py` 側に寄せる
- `orchestrator.py` の private method の並び順を
  - lifecycle
  - submit path
  - run loop
  - proactive
  - turn helpers
  にそろえる
- `docs/architecture.md` は構造が落ち着いたタイミングで更新する

## 次回の再開手順

次に自分が触るときは、最初にこれをやる。

1. `git status --short`
   - いま複数の新規ファイルが未追跡のはずなので、どこまでが未コミットか確認する
2. `pytest -q`
   - まず現状が壊れていないことを確認する
3. 次に触る塊のテストだけ追加で回す
   - proactive を触るなら `tests/unit/test_orchestrator.py`
   - playback を触るなら `tests/unit/test_playback.py`
   - request building を触るなら `tests/unit/test_request_building.py`
   - interrupt probe を触るなら `tests/unit/test_interrupt_probe.py`
   - debounce / submit を触るなら `tests/unit/test_submission.py`
4. `python3 -m compileall src tests`
   - import 崩れを最後に確認する

## 進め方の指針

- 一度に複数の責務を動かさない
- まず pure helper を抜き、次に stateful manager を抜く
- `ConversationOrchestrator` の public behavior は変えない
- `AGENT.md` に書かれている既存挙動を優先する
- 「構造をきれいにする」より「挙動を変えずに読めるようにする」を優先する

## ここで止めた理由

ここで止めたのは、次に残っている塊がどれも state とタイミングを多く持っていて、1 回の差分で無理に進めると挙動差を出しやすいから。

特に proactive と turn execution は、リファクタ自体は価値が高いが、次の 1 手としては「責務の境界を決めてから着手する」方が安全。
