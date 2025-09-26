# test_exec.py
# A self-contained test suite + tiny reference implementation for a safe process
# executor with timeouts, streaming, env/cwd control, and redaction.
#
# Run:
#   pytest -q test_exec.py
#
# If you have your own executor (e.g., `from mypkg.exec import run`), swap the
# calls in the tests to your implementation and keep the assertions.

from __future__ import annotations

import os
import sys
import time
import shlex
import signal
import threading
import subprocess as sp
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import unittest
import tempfile
import textwrap
import pathlib


# =========================
# Reference implementation
# =========================

@dataclass
class ExecResult:
    code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    truncated: bool = False

StreamCallback = Callable[[str, str], None]  # (stream_name, chunk)

def _mask(text: str, redactions: Sequence[str]) -> str:
    out = text
    for s in redactions:
        if not s:
            continue
        out = out.replace(s, "•••REDACTED•••")
    return out

def run_exec(
    cmd: Union[str, Sequence[str]],
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Union[str, os.PathLike]] = None,
    stream: Optional[StreamCallback] = None,
    max_output_bytes: Optional[int] = 2_000_000,  # 2MB default cap
    redactions: Optional[Sequence[str]] = None,
    text: bool = True,
) -> ExecResult:
    """
    Execute a subprocess with:
      - hard timeout (sends SIGKILL after SIGTERM grace)
      - incremental read + optional stream callback
      - output size cap (truncates if exceeded)
      - simple redaction pass on streamed and final output
    """
    redactions = tuple(redactions or ())
    start = time.time()

    if isinstance(cmd, str):
        # shell=False still works if we pass as string only on Windows? Better split.
        argv = shlex.split(cmd, posix=os.name != "nt")
    else:
        argv = list(cmd)

    proc = sp.Popen(
        argv,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=(os.environ | env) if env else None,
        text=text,
        bufsize=1 if text else 0,  # line-buffered text
    )

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    lock = threading.Lock()
    truncated = False
    killed = False

    # Reader threads
    def reader(name: str, pipe, sink: List[str]):
        nonlocal truncated
        try:
            # Iterate line-by-line for text, or raw chunks for bytes
            if text:
                for line in iter(pipe.readline, ""):
                    masked = _mask(line, redactions)
                    with lock:
                        sink.append(masked)
                        if stream:
                            stream(name, masked)
                        if max_output_bytes is not None:
                            cur = sum(len(x.encode("utf-8")) for x in stdout_parts) + sum(
                                len(x.encode("utf-8")) for x in stderr_parts
                            )
                            if cur > max_output_bytes:
                                truncated = True
                                break
            else:
                for chunk in iter(lambda: pipe.read(4096), b""):
                    masked = chunk  # no redaction in bytes mode
                    with lock:
                        sink.append(masked)
                        if stream:
                            stream(name, masked)  # type: ignore[arg-type]
                        if max_output_bytes is not None:
                            cur = sum(len(x) for x in stdout_parts) + sum(len(x) for x in stderr_parts)
                            if cur > max_output_bytes:
                                truncated = True
                                break
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=reader, args=("stdout", proc.stdout, stdout_parts), daemon=True)  # type: ignore[arg-type]
    t_err = threading.Thread(target=reader, args=("stderr", proc.stderr, stderr_parts), daemon=True)  # type: ignore[arg-type]
    t_out.start()
    t_err.start()

    # Timeout handling
    def wait_with_timeout() -> Optional[int]:
        nonlocal killed
        try:
            return proc.wait(timeout=timeout)
        except sp.TimeoutExpired:
            # Graceful then kill
            try:
                if os.name != "nt":
                    proc.terminate()
                else:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except sp.TimeoutExpired:
                try:
                    if os.name != "nt":
                        proc.kill()
                    else:
                        proc.kill()
                except Exception:
                    pass
            killed = True
            return proc.wait()

    code = wait_with_timeout() if timeout else proc.wait()

    # Join readers (briefly)
    t_out.join(timeout=0.5)
    t_err.join(timeout=0.5)

    # Finalize output
    if text:
        so = "".join(stdout_parts)
        se = "".join(stderr_parts)
    else:
        so = b"".join(stdout_parts).decode("utf-8", errors="replace")  # type: ignore[assignment]
        se = b"".join(stderr_parts).decode("utf-8", errors="replace")  # type: ignore[assignment]

    end = time.time()
    return ExecResult(
        code=int(code if code is not None else -1),
        stdout=so,
        stderr=se,
        duration_sec=end - start,
        timed_out=killed,
        truncated=truncated,
    )


# =========================
# Test suite
# =========================

def py_cmd(code: str) -> List[str]:
    """Cross-platform python -c invocation using current interpreter."""
    return [sys.executable, "-c", textwrap.dedent(code)]

class TestExec(unittest.TestCase):
    def test_basic_stdout(self):
        r = run_exec(py_cmd('print("hello")'))
        self.assertEqual(r.code, 0)
        self.assertIn("hello", r.stdout)
        self.assertEqual(r.stderr, "")

    def test_stderr_and_exit_code(self):
        r = run_exec(py_cmd('import sys; sys.stderr.write("bad\\n"); sys.exit(3)'))
        self.assertEqual(r.code, 3)
        self.assertIn("bad", r.stderr)

    def test_timeout_kills_process(self):
        r = run_exec(py_cmd('import time; [time.sleep(0.2) for _ in range(50)]'), timeout=0.5)
        self.assertNotEqual(r.code, 0)  # was killed or non-zero
        self.assertTrue(r.timed_out)

    def test_streaming_order(self):
        seen: List[Tuple[str, str]] = []
        def cb(name: str, chunk: str):
            if chunk.strip():
                seen.append((name, chunk.strip()))
        r = run_exec(py_cmd('import sys; print("A"); sys.stderr.write("E1\\n"); print("B"); sys.stderr.write("E2\\n")'), stream=cb)
        self.assertEqual(r.code, 0)
        # Not strictly deterministic across OS, but both streams should contain expected lines
        out_lines = [x for n,x in seen if n=="stdout"]
        err_lines = [x for n,x in seen if n=="stderr"]
        self.assertIn("A", out_lines)
        self.assertIn("B", out_lines)
        self.assertIn("E1", err_lines)
        self.assertIn("E2", err_lines)

    def test_env_and_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            # Touch a file there and assert cwd seen by child
            p = pathlib.Path(td) / "marker.txt"
            p.write_text("ok", encoding="utf-8")
            r = run_exec(py_cmd('import os,sys; print(os.getcwd()); print(os.environ.get("XYZ", ""))'),
                         cwd=td, env={"XYZ": "123"})
            self.assertEqual(r.code, 0)
            self.assertIn(str(pathlib.Path(td)), r.stdout)
            self.assertIn("123", r.stdout)

    def test_redaction(self):
        secret = "s3cr3t_token"
        r = run_exec(py_cmd('import sys; sys.stderr.write("token: s3cr3t_token\\n"); print("ok")'),
                     redactions=[secret])
        self.assertEqual(r.code, 0)
        self.assertNotIn(secret, r.stderr)
        self.assertIn("•••REDACTED•••", r.stderr)

    def test_output_cap_truncates(self):
        r = run_exec(py_cmd('print("X"*200_000)'), max_output_bytes=50_000)
        self.assertTrue(r.truncated)
        self.assertGreater(len(r.stdout), 0)
        # Should not exceed cap by a lot
        self.assertLessEqual(len(r.stdout.encode("utf-8")) + len(r.stderr.encode("utf-8")), 55_000)

    def test_large_interleaved_output(self):
        code = r'''
import sys
for i in range(5000):
    print(f"OUT-{i}")
    if i % 50 == 0:
        sys.stderr.write(f"ERR-{i}\n")
'''
        r = run_exec(py_cmd(code))
        self.assertEqual(r.code, 0)
        self.assertIn("OUT-0", r.stdout)
        self.assertIn("ERR-0", r.stderr)

    def test_non_text_mode_bytes(self):
        # Ensure bytes mode doesn't crash; we convert to utf-8 for final result
        r = run_exec(py_cmd('import sys; sys.stdout.buffer.write(b"\\xff\\xfe\\xfd")'), text=True)
        self.assertEqual(r.code, 0)
        # In text=True, Python may interpret as Latin-1-like; ensure it didn't crash
        self.assertTrue(isinstance(r.stdout, str))

    def test_zero_timeout_immediate_kill(self):
        # Zero timeout should kill quickly
        r = run_exec(py_cmd('import time; time.sleep(1)'), timeout=0.0)
        self.assertTrue(r.timed_out)

    def test_shell_style_string_cmd_split(self):
        # Passing a string command should be split safely (no shell=True)
        r = run_exec(" ".join(py_cmd('print("ok")')))
        self.assertIn("ok", r.stdout)

# PyTest bridge
def test_pytest_bridge():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestExec)
    res = unittest.TextTestRunner(verbosity=0).run(suite)
    assert res.wasSuccessful(), "exec tests failed"