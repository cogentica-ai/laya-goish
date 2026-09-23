#!/usr/bin/env python3
import json,pathlib,subprocess
root=pathlib.Path(__file__).resolve().parents[1]
model=root/"models/laya_english_f16.gguf"
texts=["","Hello, world!","I'm you're we've he'll she'd can't I'M","  word   next\n\n\tlast  ","1234567890 + 42.5","café Cafe\u0301 日本語 🙂 ไทย","[CLS][SEP][MASK][PAD][UNK]","before [MASK] after","https://example.com?a=1&b=2","\r\n\t \v\f"," a  b   c","'s 't 're 've 'm 'll 'd","\u0000\u0001\u0007 hi","The quick brown fox jumps over the lazy dog."]
p=subprocess.run([str(root/"reference-build/reference"),str(model),"tokenize"],input="".join(json.dumps(t,ensure_ascii=False)+"\n" for t in texts),text=True,capture_output=True,check=True)
expected=[json.loads(s) for s in p.stdout.splitlines()]
for t,e in zip(texts,expected):
 if "\x00" in t: continue # argv cannot carry NUL; exercised via JSON requests instead.
 p=subprocess.run([str(root/"target/x86_64-unknown-linux-gnu/release/laya"),"tokenize",str(model),"--text",t],text=True,capture_output=True,check=True)
 a=json.loads(p.stdout)
 assert a==e,(repr(t),a,e)
assert len(expected)==len(texts)
print(f"PASS: {len(texts)-1} tokenizer parity cases")
