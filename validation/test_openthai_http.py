#!/usr/bin/env python3
"""OpenThai integration through the shared Goish HTTP server."""
import concurrent.futures,json,pathlib,time
from test_http import Server,wait_busy
ROOT=pathlib.Path(__file__).resolve().parents[1]
reqs=json.loads((ROOT/"validation/openthai.requests.json").read_text())
actual=json.loads((ROOT/"validation/openthai.actual.json").read_text())
checks=[]
def record(s):checks.append(s);print("PASS",s,flush=True)
s=Server(model=ROOT/"models/openthai_systemone_v03_f16.gguf")
try:
 assert s.call("GET","/health")[1]["model"]=="openthai-systemone"
 assert [m["name"] for m in s.call("GET","/v1/models")[1]["models"]]==["openthai-systemone"]
 record("OpenThai health and model identity")
 for i,path in [(1,"/v1/systemone"),(0,"/v1/decide")]:
  status,out,_=s.call("POST",path,dict(reqs[i],id="openthai-http",model="openthai-systemone"))
  assert status==200 and out["answers"]==actual[i]["answers"] and out["id"]=="openthai-http",(status,out)
 record("Thai multi-question and English decisions match native CLI exactly")
 batch={"states":[reqs[6]["state"]],"questions":reqs[6]["questions"],"permutations":3}
 status,out,_=s.call("POST","/v1/decide/batch",batch)
 assert status==200 and out["results"][0]["answers"]==actual[6]["answers"] and out["results"][0]["usage"]["permutations"]==3
 record("batch preserves option-order averaging settings")
 for req in [
  dict(reqs[0],model="laya"),
  dict(reqs[0],permutations=33),
  dict(reqs[0],order_invariant="yes"),
  dict(reqs[0],state=42),
  {"state":"x","questions":{"q":{"type":"choice","instructions":"pick","criteria":{str(i):None for i in range(256)}}}},
  {"state":"x","questions":{"q":{"type":"score","instructions":"rate","criteria":["x"]*11}}},
  {"state":"x","questions":{"q":{"type":"noul","instructions":"word "*5000}}},
 ]:
  status,out,_=s.call("POST","/v1/decide",req);assert status==422,(status,out)
 record("model, schema, option limits and context overflow return 422")
 with concurrent.futures.ThreadPoolExecutor() as pool:
  f=pool.submit(s.call,"POST","/v1/decide",reqs[1]);wait_busy(s)
  start=time.monotonic();assert s.call("GET","/health")[0]==200
  status,_,headers=s.call("POST","/v1/decide",reqs[0])
  assert status==429 and headers["Retry-After"]=="1" and time.monotonic()-start<1
  assert f.result()[1]["answers"]==actual[1]["answers"]
 record("health stays responsive and overload returns 429 during OpenThai inference")
finally:s.close()
assert s.p.returncode==0
(ROOT/"validation/openthai-http.report.json").write_text(json.dumps(dict(status="passed",checks=checks),indent=2)+"\n")
