from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from vocalive.pipeline.interruption import CancellationToken


_DEFAULT_CSC_CANDIDATES = (
    Path(os.environ.get("WINDIR", r"C:\Windows"))
    / "Microsoft.NET"
    / "Framework64"
    / "v4.0.30319"
    / "csc.exe",
    Path(os.environ.get("WINDIR", r"C:\Windows"))
    / "Microsoft.NET"
    / "Framework"
    / "v4.0.30319"
    / "csc.exe",
)


def find_csharp_compiler() -> str | None:
    compiler = shutil.which("csc.exe") or shutil.which("csc")
    if compiler:
        return compiler
    for candidate in _DEFAULT_CSC_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


async def ensure_csharp_helper(
    *,
    source: str,
    source_path: Path,
    output_path: Path,
    references: tuple[str, ...] = (),
    timeout_seconds: float,
    build_timeout_floor_seconds: float,
    cancellation: CancellationToken | None = None,
    unavailable_message: str,
    timeout_message: str,
    failure_message: str,
) -> Path:
    compiler = find_csharp_compiler()
    if compiler is None:
        raise RuntimeError(unavailable_message)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    with tempfile.NamedTemporaryFile(
        dir=source_path.parent,
        prefix=f"{output_path.stem}-",
        suffix=output_path.suffix,
        delete=False,
    ) as handle:
        temporary_output_path = Path(handle.name)
    try:
        try:
            process = await asyncio.create_subprocess_exec(
                compiler,
                "/nologo",
                "/target:exe",
                *(f"/r:{reference}" for reference in references),
                f"/out:{temporary_output_path}",
                str(source_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await communicate_with_cancellation(
                process=process,
                cancellation=cancellation,
                timeout_seconds=max(timeout_seconds, build_timeout_floor_seconds),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(unavailable_message) from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError(timeout_message) from exc
        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
            detail_text = stderr_text or stdout_text
            detail = f": {detail_text}" if detail_text else ""
            raise RuntimeError(f"{failure_message}{detail}")
        temporary_output_path.replace(output_path)
        return output_path
    finally:
        temporary_output_path.unlink(missing_ok=True)


async def communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    cancellation: CancellationToken | None,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    communicate_task = asyncio.create_task(process.communicate())
    try:
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait(
                {communicate_task},
                timeout=min(remaining, 0.1),
            )
            if communicate_task in done:
                return await communicate_task
    except BaseException:
        await terminate_process(process)
        if not communicate_task.done():
            communicate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await communicate_task
        raise


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.kill()
    with contextlib.suppress(ProcessLookupError):
        await process.wait()
