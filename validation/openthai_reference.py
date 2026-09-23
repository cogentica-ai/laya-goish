#!/usr/bin/env python3
"""Save CPU/F32 outputs from the pinned upstream OpenThai implementation."""
import importlib.util,json,pathlib,sys,time,urllib.request
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/"upstream/openthai-systemone"
torch.set_num_threads(4)
# Reference code is test-only; pin both the checkpoint wrapper and client revision.
HF_REVISION="f3709948b5e3cc9606a57e74ba62b7a639d17dd3"
CLIENT_REVISION="5d04bcca0c58bd10e7dac2d3d369d8f760bea6cf"
for name in ["configuration.py","modeling.py","formatting.py","types.py","tokenizer_config.json"]:
    (SOURCE/name).write_bytes(urllib.request.urlopen(f"https://huggingface.co/iapp/OpenThai-SystemOne/resolve/{HF_REVISION}/{name}").read())
(SOURCE/"client.py").write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/iapp-technology/openthai-systemone/{CLIENT_REVISION}/openthai_systemone/client.py").read())

(SOURCE/"__init__.py").touch()
spec=importlib.util.spec_from_file_location("openthai_reference",SOURCE/"__init__.py",submodule_search_locations=[str(SOURCE)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from openthai_reference.client import SystemOneClient
requests=[
 {"state":"I was charged twice. Please refund the duplicate payment.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}},
 {"state":{"ticket":"ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรมาสามรอบแล้ว"},"questions":{
 "department":{"type":"choice","instructions":"ทีมใดควรรับผิดชอบ","criteria":{"billing":"การเงิน/ค่าบริการ","technical":"ระบบใช้งานไม่ได้","sales":None}},
 "frustration":{"type":"score","instructions":"ลูกค้าหงุดหงิดแค่ไหน","criteria":["ใจเย็น","หงุดหงิดแต่สุภาพ","โกรธมาก"]},
 "refund_requested":{"type":"noul","instructions":"ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"}}},
 {"state":"Cafe\u0301 🐈\n\t你好 สวัสดี ๑๒๓ 123 I'M we'd <|ts_answer|>","questions":{"q":{"type":"choice","instructions":"Which language appears?","criteria":{"Thai":None,"English":"English letters","other":"none"}}}},
 {"state":"No information.","questions":{"q":{"type":"choice","instructions":"Pick the matching option.","criteria":{"only":None}}}},
 {"state":"The invoice amount is 12.3456789.","questions":{"q":{"type":"noul","instructions":"Is there an invoice?","criteria":{"false":"There is no invoice.","true":"An invoice is present."}}}},
 {"state":"Select number 7.","questions":{"q":{"type":"choice","instructions":"Which number is requested?","criteria":{str(i):None for i in range(17)}}},"permutations":1},
 {"state":"Select the blue color.","questions":{"q":{"type":"choice","instructions":"Which color?","criteria":{"red":None,"blue":None,"green":None}}},"permutations":3},
]
(ROOT/"validation/openthai.requests.json").write_text(json.dumps(requests,ensure_ascii=False,indent=2)+"\n")
requests.extend([
 {"state":"Select number 123.","questions":{"q":{"type":"choice","instructions":"Which number?","criteria":{str(i):None for i in range(255)}}},"permutations":1},
 {"state":"Select number 7.","questions":{"q":{"type":"choice","instructions":"Which number?","criteria":{str(i):None for i in range(11)}}}},
 {"state":{"amount":12.3456789,"valid":True,"items":[1,None,"สวัสดี"]},"questions":{"q":{"type":"noul","instructions":"Is the amount greater than ten?"}}},
])
(ROOT/"validation/openthai.requests.json").write_text(json.dumps(requests,ensure_ascii=False,indent=2)+"\n")
print("Loading reference",flush=True)
from transformers import PreTrainedTokenizerFast
import openthai_reference.client as client_module
class LocalTokenizer:
 @staticmethod
 def from_pretrained(path):
  return PreTrainedTokenizerFast(tokenizer_file=str(SOURCE/"tokenizer.json"),pad_token="<|endoftext|>",eos_token="<|endoftext|>")
client_module.AutoTokenizer=LocalTokenizer
client=SystemOneClient(str(SOURCE),device="cpu",dtype=torch.float32,max_total_tokens=4096,max_state_tokens=2048)
# Match a deterministic portable eager implementation for the full-attention layers.
client.model.model.config._attn_implementation="eager"
results=[]
for i,req in enumerate(requests):
 start=time.monotonic()
 parsed={k:__import__("openthai_reference.types",fromlist=["parse_question"]).parse_question(v) for k,v in req["questions"].items()}
 enc=client.fmt.encode(req["state"],parsed)
 batch=__import__("openthai_reference.formatting",fromlist=["collate"]).collate([enc],client.pad_id)
 with torch.no_grad():
  hidden=client.model.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],use_cache=False).last_hidden_state
  raw=client.model.slot_head(client.model.gather_answer_states(hidden,batch["answer_positions"])).float()[0].flatten().tolist()
 response=client.system_one(req["state"],req["questions"],permutations=req.get("permutations")).model_dump()
 results.append(dict(input_ids=enc.input_ids,answer_positions=enc.answer_positions,raw_logits=raw,response=response))
 print("reference",i,"tokens",len(enc.input_ids),"seconds",round(time.monotonic()-start,2),flush=True)
 (ROOT/"validation/openthai.reference.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print("Reference complete",flush=True)
