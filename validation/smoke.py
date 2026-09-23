#!/usr/bin/env python3
import json,pathlib,struct,subprocess,tempfile
root=pathlib.Path(__file__).resolve().parents[1]
b=root/"target/x86_64-unknown-linux-gnu/release/laya"
m=root/"models/laya_english_f16.gguf"
def proc(args,**kwargs):
 return subprocess.run([str(b)]+list(map(str,args)),text=True,capture_output=True,timeout=120,**kwargs)
assert proc(["self-test"]).returncode==0
assert len(json.loads(proc(["list-presets"]).stdout))==10
for args in [["unknown"],["decide",m,"--threads","0"],["decide",m,"--device","cuda"],["decide",m,"--nonsense"],["decide",m,"--request"]]:
 p=proc(args);assert p.returncode!=0,(args,p.stdout,p.stderr)
with tempfile.TemporaryDirectory() as td:
 for i,blob in enumerate([b"",b"GGUF",b"GGUF"+struct.pack("<IQQ",3,999999999,0),b"NOPE"+bytes(20)]):
  f=pathlib.Path(td)/str(i);f.write_bytes(blob);assert proc(["info",f]).returncode!=0
request=json.loads((root/"examples/refund.json").read_text())
request["id"]="good-1"
second=dict(request,id="good-2")
bad=dict(request,questions={"q":{"type":"unknown"}})
lines=[json.dumps(request),"{invalid",json.dumps(bad),json.dumps(second)]
p=proc(["daemon",m,"--threads","4","--raw"],input="\n".join(lines)+"\n")
assert p.returncode==0,p.stderr
out=[json.loads(line) for line in p.stdout.splitlines()]
assert len(out)==4 and out[0]["id"]=="good-1" and out[3]["id"]=="good-2"
assert "error" in out[1] and "error" in out[2]
assert out[0]["answers"]==out[3]["answers"],"daemon results changed across identical requests"
p=proc(["decide",m,"--request",root/"examples/refund.json","--threads","1","--raw"])
one=json.loads(p.stdout)
assert one["answers"]==out[0]["answers"],"thread-count changed results"
numeric=proc(["decide",m,"--request",root/"validation/numeric.request.json","--raw"])
a=json.loads(numeric.stdout)["answers"]["refund"]
e=json.loads((root/"validation/numeric.reference.json").read_text())["refund"]
assert a["input_ids"]==e["input_ids"],"numeric state serialization mismatch"
assert max(abs(x-y) for x,y in zip(a["logits"],e["logits"]))<0.03
print("PASS: self-tests, 10 presets, invalid CLI/GGUF, daemon recovery, repeated requests, 1/4-thread parity, numeric-state parity")
