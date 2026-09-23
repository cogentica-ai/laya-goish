#!/usr/bin/env python3
"""Black-box HTTP integration tests; temporary servers are always stopped."""
import concurrent.futures, http.client, json, os, pathlib, signal, socket, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
BIN=ROOT/"laya"
MODEL=ROOT/"models/laya_english_f16.gguf"
REQ=json.loads((ROOT/"examples/refund.json").read_text())
checks=[]
def record(name): checks.append(name);print("PASS:",name,flush=True)
class Server:
 def __init__(self,key=None,model=None,extra_args=None):
  with socket.socket() as sock:
   sock.bind(("127.0.0.1",0));self.port=sock.getsockname()[1]
  env=dict(os.environ);env.pop("LAYA_API_KEY",None);env.pop("TYPESAFE_API_KEY",None)
  if key:env["LAYA_API_KEY"]=key
  self.log=open(ROOT/"validation"/f"http-{self.port}.log","w")
  self.p=subprocess.Popen([str(BIN),"serve",str(model or MODEL),"--port",str(self.port)]+list(extra_args or []),stdout=self.log,stderr=self.log,env=env)
  for _ in range(900):
   if self.p.poll() is not None:raise RuntimeError(f"server exited {self.p.returncode}; see {self.log.name}")
   try:
    if self.call("GET","/health")[0]==200:return
   except (OSError,http.client.HTTPException):pass
   time.sleep(.1)
  self.close();raise RuntimeError("server did not become ready")
 def close(self):
  if self.p.poll() is None:
   self.p.terminate()
   try:self.p.wait(timeout=40)
   except subprocess.TimeoutExpired:self.p.kill();self.p.wait();raise
  self.log.close()
 def call(self,method,path,obj=None,headers=None,raw=None,chunked=False):
  h=dict(headers or {})
  body=raw
  if obj is not None:body=json.dumps(obj).encode();h.setdefault("Content-Type","application/json")
  if chunked:body=iter([body[:17],body[17:]])
  c=http.client.HTTPConnection("127.0.0.1",self.port,timeout=90)
  try:
   c.request(method,path,body=body,headers=h,encode_chunked=chunked)
   r=c.getresponse();data=r.read();status=r.status;hs=dict(r.getheaders())
   return status,json.loads(data) if data else None,hs
  finally:c.close()
 def idle(self):
  for _ in range(600):
   if not self.call("GET","/health")[1]["busy"]:return
   time.sleep(.05)
  raise AssertionError("server remained busy")
def wait_busy(s):
 for _ in range(100):
  if s.call("GET","/health")[1]["busy"]:return
  time.sleep(.02)
 raise AssertionError("inference did not become busy")
def main():
 baseline=json.loads(subprocess.check_output([str(BIN),"decide",str(MODEL),"--request",str(ROOT/"examples/refund.json")],stderr=subprocess.DEVNULL,text=True))
 s=Server()
 try:
  code,health,_=s.call("GET","/health");assert code==200 and health["model"]=="laya" and health["device"]=="cpu"
  assert s.call("HEAD","/health")[1] is None
  assert len(s.call("GET","/v1/presets")[1])==10
  assert {m["name"] for m in s.call("GET","/v1/models")[1]["models"]}=={"laya","english","jev-latest"}
  record("health, HEAD, model aliases and 10 presets")
  for path in ["/v1/systemone","/v1/decide"]:
   code,out,headers=s.call("POST",path,dict(REQ,id="http-test",model="jev-latest"))
   assert code==200 and out["answers"]==baseline["answers"] and out["id"]=="http-test",(code,out)
   assert out["usage"]["input_tokens"]==baseline["usage"]["input_tokens"]
   assert "X-Response-Time-Ms" in headers and "X-Request-Id" in headers
  record("both decision endpoints match CLI exactly, including alias and request id")
  code,out,_=s.call("POST","/v1/decide",REQ,chunked=True)
  assert code==200 and out["answers"]==baseline["answers"]
  record("chunked JSON request")
  for method,path,kwargs,expected in [
   ("GET","/missing",{},404),("GET","/v1/decide",{},405),("POST","/health",{},405),
   ("POST","/v1/decide",{"raw":b"{}"},415),
   ("POST","/v1/decide",{"raw":b"{bad","headers":{"Content-Type":"application/json"}},400),
   ("POST","/v1/decide",{"raw":b'{"x":1,"x":2}',"headers":{"Content-Type":"application/json"}},400),
   ("POST","/v1/decide",{"obj":[]},422),
   ("POST","/v1/decide",{"obj":dict(REQ,model="missing")},422),
   ("POST","/v1/decide",{"obj":dict(REQ,questions={"q":{"type":"invalid"}})},422),
   ("POST","/v1/decide/batch",{"obj":{"states":[],"questions":REQ["questions"]}},422),
   ("POST","/v1/decide",{"raw":b"x"*(1024*1024+1),"headers":{"Content-Type":"application/json"}},413),
   ("POST","/v1/decide",{"raw":b"x"*(1024*1024+1),"headers":{"Content-Type":"application/json"},"chunked":True},413),
  ]:
   code,out,_=s.call(method,path,**kwargs);assert code==expected,(method,path,code,out,expected)
   assert "error" in out
  record("JSON errors, methods, content type, model/schema validation and fixed/chunked 1 MiB limit")
  with concurrent.futures.ThreadPoolExecutor() as pool:
   future=pool.submit(s.call,"POST","/v1/decide",REQ)
   wait_busy(s);start=time.monotonic()
   assert s.call("GET","/health")[0]==200
   code,_,headers=s.call("POST","/v1/decide",REQ)
   assert code==429 and headers["Retry-After"]=="1"
   assert time.monotonic()-start<1,"health/busy response blocked on inference"
   assert future.result()[0]==200
  record("health stays responsive during inference; concurrent inference returns 429")
  code,out,_=s.call("POST","/v1/decide/batch",{"states":[REQ["state"],REQ["state"]],"questions":REQ["questions"]})
  assert code==200 and len(out["results"])==2
  assert all(r["answers"]==baseline["answers"] for r in out["results"])
  record("batch matches individual decisions")
  # Disconnect must stop a multi-question request between encoder forwards.
  many={"state":REQ["state"],"questions":{str(i):REQ["questions"]["refund"] for i in range(12)}}
  body=json.dumps(many).encode()
  c=socket.create_connection(("127.0.0.1",s.port));c.sendall(b"POST /v1/decide HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "+str(len(body)).encode()+b"\r\n\r\n"+body)
  wait_busy(s);c.close();start=time.monotonic();s.idle()
  assert time.monotonic()-start<8,"disconnect did not cancel between questions"
  record("client disconnect cancels remaining questions")
  with concurrent.futures.ThreadPoolExecutor() as pool:
   future=pool.submit(s.call,"POST","/v1/decide",REQ)
   wait_busy(s);s.p.send_signal(signal.SIGTERM)
   assert future.result()[0]==200,"shutdown dropped active request"
   assert s.p.wait(timeout=40)==0
  record("SIGTERM drains active inference and exits cleanly")
 finally:s.close()
 s=Server("integration-test-token")
 try:
  assert s.call("GET","/health")[0]==200 and s.call("GET","/v1/presets")[0]==200
  assert s.call("GET","/v1/models")[0]==401
  assert s.call("POST","/v1/decide",REQ)[0]==401
  assert s.call("POST","/v1/decide",REQ,headers={"Authorization":"Bearer wrong"})[0]==401
  h={"Authorization":"Bearer integration-test-token"}
  assert s.call("GET","/v1/models",headers=h)[0]==200
  code,out,_=s.call("POST","/v1/decide",REQ,headers=h)
  assert code==200 and out["answers"]==baseline["answers"]
  record("optional bearer authentication")
 finally:s.close()
 assert s.p.returncode==0
 (ROOT/"validation/http.report.json").write_text(json.dumps({"status":"passed","checks":checks},indent=2)+"\n")
if __name__=="__main__":main()
