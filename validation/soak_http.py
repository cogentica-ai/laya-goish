#!/usr/bin/env python3
"""Bounded soak test for a running local Laya HTTP server; records /proc memory."""
import argparse, collections, concurrent.futures, http.client, json, pathlib, threading, time
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--http-only", action="store_true", help="100,000 mixed HTTP requests without inference")
args = parser.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[1]
PID = int((ROOT / "server.pid").read_text())
PROC = pathlib.Path("/proc") / str(PID)
REQ = (ROOT / "examples/refund.json").read_bytes()
OVERSIZE = b"x" * (1024 * 1024 + 1)
REPORT = ROOT / ("validation/http-churn.report.json" if args.http_only else "validation/http-soak.report.json")
stop = threading.Event()
lock = threading.Lock()
counts = collections.Counter()
samples, checkpoints, errors = [], [], []
started = time.monotonic()
phase = "baseline"
finished = False
def fields(path):
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition(":")
        if not k or not k[0].isalpha(): continue
        try: out[k] = int(v.split()[0])
        except (ValueError, IndexError): pass
    return out
def snapshot(full=False):
    s = fields(PROC / "status")
    m = fields(pathlib.Path("/proc/meminfo"))
    out = dict(elapsed_s=round(time.monotonic()-started, 2), phase=phase,
               rss_kib=s["VmRSS"], anon_kib=s["RssAnon"], file_kib=s["RssFile"],
               swap_kib=s.get("VmSwap",0), threads=s["Threads"],
               fds=len(list((PROC/"fd").iterdir())), available_kib=m["MemAvailable"])
    if full: out["smaps_rollup"] = fields(PROC/"smaps_rollup")
    return out
def monitor():
    while not stop.wait(1):
        try:
            s = snapshot()
            samples.append(s)
            if s["available_kib"] < 700*1024 or s["rss_kib"] > baseline["rss_kib"] + 3*1024*1024:
                errors.append("Memory safety threshold reached; stopping load")
                stop.set()
        except Exception as e:
            errors.append(repr(e)); stop.set()
def checkpoint(label):
    s = snapshot(True); s["label"] = label; checkpoints.append(s)
    print(json.dumps({k:v for k,v in s.items() if k!="smaps_rollup"}), flush=True)
def call(method, path, body=None, expected=(200,), connection=None):
    if stop.is_set(): raise RuntimeError("load stopped")
    c = connection or http.client.HTTPConnection("127.0.0.1",8080,timeout=60)
    try:
        c.request(method,path,body=body,headers={"Content-Type":"application/json"} if body else {})
        r=c.getresponse(); payload=r.read()
        with lock: counts[str(r.status)] += 1
        if r.status not in expected: raise AssertionError((path,r.status,payload[:300]))
        obj=json.loads(payload) if payload else None
        if path in ("/v1/decide","/v1/systemone") and r.status==200:
            assert obj["answers"]==answer, ("inference changed",obj)
        return obj
    finally:
        if connection is None: c.close()
def mixed_worker(n, persistent):
    c=http.client.HTTPConnection("127.0.0.1",8080,timeout=30) if persistent else None
    routes=[("GET","/health",None,(200,)),("GET","/v1/models",None,(200,)),
            ("GET","/v1/presets",None,(200,)),("GET","/missing",None,(404,)),
            ("POST","/v1/decide",b"{bad",(400,)),("POST","/v1/decide",b"{}",(422,))]
    try:
        for i in range(n):
            if stop.is_set(): break
            call(*routes[i%len(routes)],connection=c)
    finally:
        if c: c.close()
def parallel(fn, args, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(fn,*a) for a in args]
        for f in fs: f.result()
def save():
    REPORT.write_text(json.dumps(dict(pid=PID,elapsed_s=round(time.monotonic()-started,2),
        status=("failed" if errors else "passed" if finished else "running"),counts=dict(counts),errors=errors,
        checkpoints=checkpoints,samples=samples),indent=2)+"\n")
baseline=snapshot(True)
checkpoint("initial")
thread=threading.Thread(target=monitor,daemon=True); thread.start()
try:
    # Establish correctness independently, then check every successful HTTP inference.
    import subprocess
    if not args.http_only:
        answer=json.loads(subprocess.check_output([str(ROOT/"laya"),"decide",str(ROOT/"models/laya_english_f16.gguf"),
        "--request",str(ROOT/"examples/refund.json")],stderr=subprocess.DEVNULL))["answers"]
    phase="warmup"
    for _ in range(0 if args.http_only else 5): call("POST","/v1/decide",REQ)
    checkpoint("warm")
    for round_no in range(1,6):
        phase=f"round-{round_no}-inference"
        for _ in range(0 if args.http_only else 20): call("POST","/v1/decide",REQ)
        checkpoint(f"round-{round_no}-inference")
        phase=f"round-{round_no}-http"
        parallel(mixed_worker,[(1250 if args.http_only else 250,round_no%2==0)]*16,16)
        phase=f"round-{round_no}-concurrent-inference"
        if not args.http_only:
            parallel(call,[("POST","/v1/decide",REQ,(200,429))]*32,8)
        phase=f"round-{round_no}-oversized"
        parallel(call,[("POST","/v1/decide",OVERSIZE,(413,))]*32,8)
        time.sleep(2)
        checkpoint(f"round-{round_no}-end")
        save()
    phase="idle"
    for i in range(3):
        time.sleep(10);checkpoint(f"idle-{(i+1)*10}s")
    call("GET","/health")
    finished = True
except Exception as e:
    errors.append(repr(e));print("FAIL",repr(e),flush=True)
finally:
    stop.set();thread.join(timeout=2);save()
    print(json.dumps(dict(report=str(REPORT),counts=dict(counts),errors=errors)),flush=True)
if errors: raise SystemExit(1)
