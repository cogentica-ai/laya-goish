#!/usr/bin/env python3
"""Differential checks against unmodified upstream kernels and preprocessing."""
import argparse,json,pathlib,subprocess,time,math
ROOT=pathlib.Path(__file__).resolve().parents[1]
BIN=ROOT/"target/x86_64-unknown-linux-gnu/release/laya"
REF=ROOT/"reference-build/reference"
def run(cmd,**kw):
    p=subprocess.run([str(x) for x in cmd],text=True,capture_output=True,timeout=600,**kw)
    if p.returncode: raise RuntimeError(f"{cmd}: {p.stderr[-3000:]}")
    return json.loads(p.stdout)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="models/laya_english_f16.gguf");args=ap.parse_args()
    model=ROOT/args.model
    cases={
      "refund":{"state":"I was charged twice. Please refund the duplicate payment.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}},
      "types":{"state":{"message":"The service is down and we cannot process any orders. Fix it now.","count":2},"questions":{
        "category":{"type":"choice","instructions":"Which team should handle this request?","criteria":{"technical":"bugs, outages, system errors","billing":"invoices, payments, refunds","sales":"pricing, new contracts"}},
        "urgency":{"type":"score","instructions":"How urgent is this request?","criteria":["no time pressure","needs attention soon","blocking issue or hard deadline"]},
        "refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}},
      "unicode":{"state":"Cafe\u0301 — café. 日本語: 返金してください! 🙂\nI'm NOT asking for a refund. [MASK]","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back? [MASK]"}}},
      "truncation":{"state":"Please refund my payment. "*220,"questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}},
    }
    records=[]
    for name,req in cases.items():
        path=ROOT/"validation"/(name+".request.json");path.write_text(json.dumps(req,ensure_ascii=False))
        actual=run([BIN,"decide",model,"--request",path,"--raw"])
        expected=run([REF,model,path])
        rec={"case":name,"input_tokens":actual["usage"]["input_tokens"],"latency_ms":actual["usage"]["latency_ms"],"questions":[]}
        for q,a in actual["answers"].items():
            e=expected[q]
            assert a["input_ids"]==e["input_ids"],f"{name}/{q}: token mismatch"
            assert a["markers"]==e["markers"],f"{name}/{q}: marker mismatch"
            ld=max(abs(x-y) for x,y in zip(a["logits"],e["logits"]))
            ad=max(abs(x-y) for x,y in zip(a["act_logits"],e["act_logits"]))
            temp_by_options={"choice:2":1.9063563346862793,"choice:3-5":1.7601518630981445,"choice:6-10":1.0000158548355103,"choice:11+":0.10058280825614929,"score:3-5":1.2514300346374512,"noul:2":1.983399510383606}
            k=len(e["logits"]);kind=req["questions"][q]["type"]
            bucket="2" if k<=2 else "3-5" if k<=5 else "6-10" if k<=10 else "11+"
            temp=temp_by_options.get(kind+":"+bucket,{"choice":1.6369030475616455,"score":1.2514300346374512,"noul":1.983399510383606}[kind])
            z=[(v-max(e["logits"]))/temp for v in e["logits"]];p=[math.exp(v) for v in z];p=[v/sum(p) for v in p]
            pd=max(abs(x-y) for x,y in zip(a["probabilities"].values(),p))
            assert pd<0.025,f"{name}/{q}: probability mismatch {pd}"
            confidence=max(p) if kind=="noul" else 1+sum(v*math.log(max(v,1e-12)) for v in p)/math.log(len(p))
            assert abs(a["confidence"]-confidence)<0.04,f"{name}/{q}: confidence mismatch"
            if kind=="choice":
                keys=list(req["questions"][q]["criteria"])
                assert a["choice"]==keys[max(range(k),key=lambda i:p[i])],f"{name}/{q}: choice mismatch"
            if kind=="score":
                assert abs(a["score"]-sum(i*v for i,v in enumerate(p)))<0.04,f"{name}/{q}: score mismatch"
            action=1/(1+math.exp(max(-700,min(700,e["act_logits"][1]-e["act_logits"][0]))))
            assert abs(a["action"]["act_probability"]-action)<0.025,f"{name}/{q}: action probability mismatch"
            assert ad/max(1,max(abs(v) for v in e["act_logits"]))<0.02,f"{name}/{q}: action logits differ"
            rec["questions"].append({"id":q,"max_logit_error":ld,"max_probability_error":pd,"max_action_logit_error":ad})
            # GGML quantizes activations for quantized GEMM; Rust accumulates F32.
            assert ld<(0.03 if "_f16." in str(model) else 0.15),f"{name}/{q}: logits differ {ld}"
        records.append(rec)
        print(json.dumps(rec),flush=True)
    report={"model":args.model,"cases":records}
    (ROOT/"validation"/(model.stem+".report.json")).write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__": main()
