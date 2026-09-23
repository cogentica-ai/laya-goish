#!/usr/bin/env python3
"""Real-model routing, lifecycle and compatibility checks for a multi-model server."""
import concurrent.futures,json,pathlib,signal,struct,subprocess,tempfile,time
from test_http import Server,wait_busy
ROOT=pathlib.Path(__file__).resolve().parents[1]
BIN=ROOT/"laya"
LAYA=ROOT/"models/laya_english_f16.gguf"
THAI=ROOT/"models/openthai_systemone_v03_f16.gguf"
refund=json.loads((ROOT/"examples/refund.json").read_text())
thai=json.loads((ROOT/"examples/openthai-thai.json").read_text())
checks=[]
def record(s):checks.append(s);print("PASS",s,flush=True)
def cli(model,req):
 p=subprocess.run([str(BIN),"decide",str(model),"--request",str(ROOT/req)],capture_output=True,text=True,timeout=120,check=True)
 return json.loads(p.stdout)
def failure(args,message):
 p=subprocess.run([str(BIN)]+list(map(str,args)),capture_output=True,text=True,timeout=20)
 assert p.returncode!=0 and message in p.stderr,(p.returncode,p.stderr)
 assert "loaded " not in p.stderr,"configuration failed after loading weights"
# Startup errors must be rejected before loading weights or opening a listener.
for args,message in [
 (["serve",LAYA,"--model",LAYA],"duplicate model"),
 (["serve",LAYA,"--model",ROOT/"models/laya_english_q8_0.gguf"],"duplicate model"),
 (["serve",LAYA,"--default-model","missing"],"unknown default model"),
 (["serve",LAYA,"--model",ROOT/"models/not-present.gguf"],"open "),
 (["serve",LAYA,"--model"],"missing value"),
 (["decide",LAYA,"--model",THAI],"only supported by serve"),
 (["daemon",LAYA,"--default-model","laya"],"only supported by serve"),
]:
 failure(args,message)
with tempfile.TemporaryDirectory() as td:
 path=pathlib.Path(td)/"unsupported.gguf"
 def string(s):b=s.encode();return struct.pack("<Q",len(b))+b
 path.write_bytes(b"GGUF"+struct.pack("<IQQ",3,0,1)+string("general.architecture")+struct.pack("<I",8)+string("unsupported"))
 failure(["serve",path],"unsupported GGUF architecture")
record("duplicate identities, default selection, unreadable paths and unsupported architectures fail before weight loading")
laya=cli(LAYA,"examples/refund.json")
openthai=cli(THAI,"examples/openthai-thai.json")
record("single-model CLI baselines")
s=Server(model=LAYA,extra_args=["--model",str(THAI)])
try:
 code,health,_=s.call("GET","/health")
 assert code==200 and health["model"]=="laya" and health["models"]==["laya","openthai-systemone"]
 code,listing,_=s.call("GET","/v1/models")
 byname={m["name"]:m for m in listing["models"]}
 assert set(byname)=={"laya","english","jev-latest","openthai-systemone"}
 assert listing["default_model"]=="laya"
 assert byname["english"]["canonical_name"]==byname["jev-latest"]["canonical_name"]=="laya"
 assert byname["openthai-systemone"]["architecture"]=="openthai_systemone"
 record("both models loaded; identities, aliases and default listed")
 for name in [None,"openthai-systemone","english","openthai-systemone","jev-latest","laya"]:
  req=thai if name=="openthai-systemone" else refund
  expected=openthai if name=="openthai-systemone" else laya
  body=dict(req,id="routed")
  if name is not None:body["model"]=name
  code,out,_=s.call("POST","/v1/systemone",body)
  assert code==200 and out["model"]==expected["model"] and out["answers"]==expected["answers"] and out["id"]=="routed",(name,code,out)
 record("alternating backends, default and aliases match CLI without state contamination")
 # Each backend applies its own schema even while sharing one listener.
 many={"state":"Select number 7.","questions":{"q":{"type":"choice","instructions":"Which number?","criteria":{str(i):None for i in range(17)}}},"permutations":1}
 assert s.call("POST","/v1/decide",dict(many,model="laya"))[0]==422
 code,out,_=s.call("POST","/v1/decide",dict(many,model="openthai-systemone"))
 assert code==200 and out["usage"]["permutations"]==1
 for name in ["missing","",42,[],False]:
  assert s.call("POST","/v1/decide",dict(refund,model=name))[0]==422
 for body in [[],None,42]:
  assert s.call("POST","/v1/decide",raw=json.dumps(body).encode(),headers={"Content-Type":"application/json"})[0]==422
 record("routing errors and model-specific limits return 422")
 for name,req,expected in [("english",refund,laya),("openthai-systemone",thai,openthai)]:
  code,out,_=s.call("POST","/v1/decide/batch",{"model":name,"states":[req["state"],req["state"]],"questions":req["questions"],"permutations":1})
  assert code==200 and len(out["results"])==2
  assert all(r["model"]==expected["model"] and r["answers"]==expected["answers"] for r in out["results"])
 record("batch uses selected backend and preserves result order")
 with concurrent.futures.ThreadPoolExecutor() as pool:
  future=pool.submit(s.call,"POST","/v1/decide",dict(thai,model="openthai-systemone"))
  wait_busy(s)
  start=time.monotonic()
  code,_,headers=s.call("POST","/v1/decide",dict(refund,model="laya"))
  assert code==429 and headers["Retry-After"]=="1"
  assert s.call("GET","/health")[0]==200 and time.monotonic()-start<1
  assert future.result()[1]["answers"]==openthai["answers"]
 record("global overload protection covers cross-model requests; health remains responsive")
 with concurrent.futures.ThreadPoolExecutor() as pool:
  future=pool.submit(s.call,"POST","/v1/decide",dict(thai,model="openthai-systemone"))
  wait_busy(s);s.p.send_signal(signal.SIGTERM)
  assert future.result()[0]==200 and s.p.wait(timeout=40)==0
 record("shutdown drains inference with both backends loaded")
finally:s.close()
# Configure the second model as the default, and enforce authentication for both.
s=Server(key="registry-test-key",model=LAYA,extra_args=["--model",str(THAI),"--default-model","openthai-systemone"])
try:
 assert s.call("GET","/health")[1]["model"]=="openthai-systemone"
 assert s.call("GET","/v1/models")[0]==401
 for name in ["laya","openthai-systemone"]:
  assert s.call("POST","/v1/decide",dict(refund,model=name))[0]==401
 headers={"Authorization":"Bearer registry-test-key"}
 assert s.call("GET","/v1/models",headers=headers)[1]["default_model"]=="openthai-systemone"
 code,out,_=s.call("POST","/v1/decide",thai,headers=headers)
 assert code==200 and out["model"]=="openthai-systemone" and out["answers"]==openthai["answers"]
 code,out,_=s.call("POST","/v1/decide",dict(refund,model="jev-latest"),headers=headers)
 assert code==200 and out["model"]=="laya" and out["answers"]==laya["answers"]
 record("explicit default model and shared bearer authentication")
finally:s.close()
assert s.p.returncode==0
(ROOT/"validation/registry.report.json").write_text(json.dumps({"status":"passed","checks":checks},indent=2)+"\n")
