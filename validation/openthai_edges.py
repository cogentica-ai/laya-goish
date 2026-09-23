#!/usr/bin/env python3
"""Validate source-to-GGUF reproducibility, truncation, state reset and thread parity."""
import importlib.util,json,pathlib,subprocess,sys,hashlib
import gguf
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
model=ROOT/"models/openthai_reproduction.gguf"
if not model.exists():
 subprocess.run([sys.executable,str(ROOT/"scripts/convert-openthai.py"),"--output",str(model),
  "--report",str(ROOT/"validation/openthai-reproduction.report.json")],cwd=ROOT,check=True)
report=json.loads((ROOT/"validation/openthai-conversion.report.json").read_text())
with model.open("rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()==report["sha256"]
source=ROOT/"upstream/openthai-systemone"
spec=importlib.util.spec_from_file_location("openthai_edge_reference",source/"__init__.py",submodule_search_locations=[str(source)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from openthai_edge_reference.formatting import Formatter
from openthai_edge_reference.types import parse_question
class Tok:
 unk_token_id=None
 def __init__(self):self.t=Tokenizer.from_file(str(source/"tokenizer.json"))
 def get_vocab(self):return self.t.get_vocab()
 def convert_tokens_to_ids(self,x):
  return [self.t.token_to_id(v) for v in x] if isinstance(x,list) else self.t.token_to_id(x)
 def __call__(self,x,**kwargs):return {"input_ids":self.t.encode(x,add_special_tokens=False).ids}
try:
 # Change only the dedicated test copy's context metadata; weights remain identical.
 reader=gguf.GGUFReader(model,mode="r+")
 reader.fields["openthai.context_length"].parts[-1][0]=64
 reader.fields["openthai.state_length"].parts[-1][0]=32
 reader.data.flush()
 del reader
 base=json.loads((ROOT/"examples/refund.json").read_text())
 long=dict(base,state="obsolete "*150+"ล่าสุด ลูกค้าขอเงินคืน")
 expected=Formatter(Tok(),max_total_tokens=64,max_state_tokens=32).encode(long["state"],{k:parse_question(v) for k,v in long["questions"].items()})
 assert expected.truncated_state
 results=[]
 for threads in [1,4]:
  payload="\n".join(json.dumps(r,ensure_ascii=False) for r in [base,long,base])+"\n"
  proc=subprocess.run([str(ROOT/"laya"),"daemon",str(model),"--raw","--threads",str(threads)],input=payload,text=True,capture_output=True,timeout=120,check=True)
  rows=[json.loads(line) for line in proc.stdout.splitlines()]
  assert len(rows)==3
  assert rows[0]["answers"]==rows[2]["answers"]
  assert rows[0]["_raw"]==rows[2]["_raw"],"request state leaked"
  assert rows[1]["_raw"][0]["input_ids"]==expected.input_ids
  assert rows[1]["_raw"][0]["answer_positions"]==expected.answer_positions
  assert rows[1]["usage"]["truncated_state"] is True
  results.append(rows)
 assert [r["answers"] for r in results[0]]==[r["answers"] for r in results[1]]
 assert [r["_raw"] for r in results[0]]==[r["_raw"] for r in results[1]]
 checks=["GGUF reproduction SHA-256 identical","truncated state token IDs and answer positions match upstream",
 "repeated request after different input is bitwise identical","one/four-thread raw logits and answers are bitwise identical"]
 print("\n".join("PASS "+s for s in checks))
 (ROOT/"validation/openthai-edge.report.json").write_text(json.dumps(dict(status="passed",checks=checks),indent=2)+"\n")
finally:
 model.unlink()
