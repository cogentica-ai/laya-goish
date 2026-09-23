#!/usr/bin/env python3
"""Check exact inference repeatability while SIGURG interrupts SIMD workers."""
import argparse
import ctypes
import errno
import json
import pathlib
import platform
import select
import signal
import subprocess
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def stable(response):
    response = dict(response)
    assert "error" not in response, response
    response["usage"] = dict(response["usage"])
    response["usage"].pop("latency_ms", None)
    return response


def request(process, body):
    process.stdin.write(json.dumps(body, ensure_ascii=False) + "\n")
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 120)[0]:
        raise TimeoutError("inference did not finish within 120 seconds")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"daemon exited: {process.poll()}")
    return stable(json.loads(line))


def run(binary, model, body, threads, repeats, expected=None, interrupt=False):
    stop = threading.Event()
    sent = [0]
    errors = []
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [str(binary), "daemon", str(model), "--threads", str(threads), "--raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
        )

        def signals():
            try:
                while not stop.is_set():
                    # tgkill is 234 on Linux x86-64. Target all current workers;
                    # process-directed kill alone may only reach the supervisor.
                    for task in pathlib.Path(f"/proc/{process.pid}/task").iterdir():
                        result = libc.syscall(234, process.pid, int(task.name), signal.SIGURG)
                        if result == 0:
                            sent[0] += 1
                        elif ctypes.get_errno() != errno.ESRCH:
                            raise OSError(ctypes.get_errno(), "tgkill failed")
                    stop.wait(0.005)
            except Exception as error:
                errors.append(error)

        pump = None
        completed = False
        try:
            # Finish loading before sending signals. The first response also
            # verifies that thread-count changes do not change any raw output.
            first = request(process, body)
            if expected is None:
                expected = first
            assert first == expected, "baseline/thread-count mismatch"
            if interrupt:
                pump = threading.Thread(target=signals, daemon=True)
                pump.start()
            for _ in range(repeats):
                assert request(process, body) == expected, "preemption changed inference output"
            assert not errors, errors
            if interrupt:
                assert sent[0] > 0, "no SIGURG signals delivered"
            completed = True
            return expected, sent[0]
        finally:
            stop.set()
            if pump is not None:
                pump.join(timeout=5)
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
            if completed:
                assert process.returncode == 0, "daemon failed during shutdown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--baseline", type=pathlib.Path,
                        help="optional previous binary for compatibility comparison")
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "validation/goish-alpha14-preemption.report.json")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this test requires Linux x86-64")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    binary = ROOT / "target/x86_64-unknown-linux-gnu/release/laya"
    cases = [
        ("laya_english_f16.gguf", "refund.json"),
        ("laya_english_q8_0.gguf", "refund.json"),
        ("laya_english_ud_q4_k_m.gguf", "refund.json"),
        ("openthai_systemone_v03_f16.gguf", "openthai-thai.json"),
    ]
    rows = []
    for model_name, request_name in cases:
        model = ROOT / "models" / model_name
        body = json.loads((ROOT / "examples" / request_name).read_text())
        if model_name.startswith("openthai"):
            body["permutations"] = 1
        started = time.monotonic()
        expected, _ = run(args.baseline or binary, model, body, 1, 0)
        for threads in (1, 4):
            _, sent = run(binary, model, body, threads, args.repeats, expected, True)
            row = dict(model=model_name, threads=threads, repetitions=args.repeats,
                       signals_delivered=sent, exact_output_match=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
        print(f"PASS {model_name}: {time.monotonic() - started:.1f}s", flush=True)
    args.output.write_text(json.dumps(dict(
        status="passed", compared_to_previous_binary=args.baseline is not None,
        signal="SIGURG", cases=rows,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
