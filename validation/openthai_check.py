#!/usr/bin/env python3
"""Differential checks for GGUF OpenThai against saved upstream CPU/F32 outputs."""
import json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def numbers(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys(), (a.keys(),b.keys())
  return max([numbers(v,b[k]) for k,v in a.items()]+[0.])
 if isinstance(a,(float,int)) and not isinstance(a,bool):return abs(float(a)-float(b))
 assert a==b,(a,b)
 return 0.
def main():
 reqs=json.loads((ROOT/"validation/openthai.requests.json").read_text())
 refs=json.loads((ROOT/"validation/openthai.reference.json").read_text())
 rows=[];outputs=[];fail=[]
 log=(ROOT/"validation/openthai-daemon.log").open("w")
 p=subprocess.Popen([str(ROOT/"laya"),"daemon",str(ROOT/"models/openthai_systemone_v03_f16.gguf"),"--raw"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
 try:
  for i,(req,ref) in enumerate(zip(reqs,refs)):
   start=time.monotonic();p.stdin.write(json.dumps(req,ensure_ascii=False)+"\n");p.stdin.flush()
   line=p.stdout.readline()
   if not line:raise RuntimeError(f"daemon exited: {p.poll()}")
   out=json.loads(line);outputs.append(out)
   if "error" in out:raise RuntimeError(out)
   raw=out["_raw"][0]
   assert len(raw["logits"])==len(ref["raw_logits"])
   delta=max(abs(a-b) for a,b in zip(raw["logits"],ref["raw_logits"]))
   prob=numbers(out["answers"],ref["response"]["answers"])
   exact=raw["input_ids"]==ref["input_ids"] and raw["answer_positions"]==ref["answer_positions"]
   choices=all(v.get("choice")==ref["response"]["answers"][q].get("choice") for q,v in out["answers"].items())
   passed=exact and choices and delta<0.002 and prob<0.001 and out["usage"]["input_tokens"]==ref["response"]["usage"]["input_tokens"] and out["usage"]["permutations"]==ref["response"]["usage"]["permutations"]
   row=dict(case=i,tokens=out["usage"]["input_tokens"],permutations=out["usage"]["permutations"],tokens_exact=exact,choices_exact=choices,max_logit_error=delta,max_answer_error=prob,elapsed_s=round(time.monotonic()-start,3),passed=passed)
   rows.append(row)
   if not passed:fail.append(i)
   print(json.dumps(row),flush=True)
 finally:
  p.stdin.close()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  log.close()
 (ROOT/"validation/openthai.actual.json").write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+"\n")
 (ROOT/"validation/openthai.report.json").write_text(json.dumps(dict(status="passed" if not fail else "failed",reference="upstream CPU float32, transformers 5.17.0",cases=rows),indent=2)+"\n")
 if fail:raise SystemExit(f"Failed cases: {fail}")
if __name__=="__main__":main()
