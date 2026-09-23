#!/usr/bin/env python3
"""Compare the native Qwen tokenizer to the published Hugging Face tokenizer."""
import json,pathlib,random,subprocess
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
tok=Tokenizer.from_file(str(ROOT/"upstream/openthai-systemone/tokenizer.json"))
rng=random.Random(7)
samples=[
 "", "สวัสดีครับ ภาษาไทย ๑๒๓", "Hello, world! I'M we'd 12345",
 "Cafe\u0301 e\u0301\t\r\n hello  ", "你好，日本語 🐈‍⬛ 👨‍👩‍👧‍👦",
 "<|ts_state|> เรื่อง\n<|ts_q|><|ts_noul|> ใช่ไหม\n<|ts_opt_0|> no\n<|ts_opt_1|> yes\n<|ts_answer|>\n",
 "'ſabc 'Sabc 'REabc 'Veabc 'LLabc",
 "a\u0085b a\u00a0b a\u2003b a\u2028b a\u2029b\r\n \t",
 "\n".join("".join(rng.choice("abcABC0123456789 ไทยกิ้่ำๆ漢字é\u0301\u200b!?' \\t\n🐈") for _ in range(rng.randrange(1,80))) for _ in range(100)),
]
rows=[]
for i,s in enumerate(samples):
 actual=json.loads(subprocess.check_output([str(ROOT/"laya"),"tokenize",str(ROOT/"models/openthai_systemone_v03_f16.gguf"),"--text",s],text=True))
 expected=tok.encode(s,add_special_tokens=False).ids
 row=dict(case=i,token_count=len(expected),exact=actual==expected)
 rows.append(row);print(row,flush=True)
 if actual!=expected:print("ACTUAL",actual,"EXPECTED",expected,flush=True)
(ROOT/"validation/openthai-tokenizer.report.json").write_text(json.dumps(dict(status="passed" if all(r["exact"] for r in rows) else "failed",cases=rows),indent=2)+"\n")
assert all(r["exact"] for r in rows)
