#!/usr/bin/env python3
"""Convert OpenThai-SystemOne safetensors to the GGUF v3 consumed by laya-goish."""
import argparse, hashlib, json, pathlib, urllib.request
import numpy as np
import gguf

REPO = "iapp/OpenThai-SystemOne"
REVISION = "f3709948b5e3cc9606a57e74ba62b7a639d17dd3"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=pathlib.Path,default=pathlib.Path("upstream/openthai-systemone"))
    p.add_argument("--output",type=pathlib.Path,default=pathlib.Path("models/openthai_systemone_v03_f16.gguf"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--report",type=pathlib.Path,default=pathlib.Path("validation/openthai-conversion.report.json"))
    p.add_argument("--context",type=int,default=4096,help="CPU runtime token limit (32..8192)")
    a=p.parse_args()
    if not 32<=a.context<=8192:p.error("context must be 32..8192")
    a.source.mkdir(parents=True,exist_ok=True)
    if a.download:
        for name in ["config.json","tokenizer.json","model.safetensors"]:
            dest=a.source/name
            if not dest.exists():
                tmp=dest.with_suffix(dest.suffix+".part")
                print("Downloading",name,flush=True)
                urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",tmp)
                tmp.replace(dest)
    expected = {
        "model.safetensors": "e5bf18fbd975eb55de3ae8dd328f6e48263fea9d3e47b7cfe52306e98ae5f55a",
        "tokenizer.json": "d3400f1532d54f98d82b24c503e61908aa66f84e22e566e59fc80494add3a5c3",
    }
    for name,digest in expected.items():
        with (a.source/name).open("rb") as source_file:
            if hashlib.file_digest(source_file,"sha256").hexdigest()!=digest:
                raise ValueError(f"{name}: checksum mismatch against pinned v0.3 revision")
    data=(a.source/"config.json").read_bytes()
    if hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!="d03bf878f6563dd505c0a739ce9b7cc1d3bed63b":
        raise ValueError("config.json: pinned revision mismatch")
    cfg=json.loads(data)
    if cfg["model_type"]!="openthai_systemone":raise ValueError("not an OpenThai-SystemOne checkpoint")
    tok=json.loads((a.source/"tokenizer.json").read_text())
    if tok["normalizer"]!={"type":"NFC"} or tok["model"]["type"]!="BPE":raise ValueError("unsupported tokenizer")
    vocab=tok["model"]["vocab"]
    n=cfg["text_config"]["vocab_size"]
    tokens=[""]*n;types=[1]*n
    for token,i in vocab.items():tokens[i]=token
    for t in tok["added_tokens"]:
        if t["single_word"] or t["lstrip"] or t["rstrip"] or t["normalized"]:raise ValueError("unsupported added-token flags")
        tokens[t["id"]]=t["content"];types[t["id"]]=3 if t["special"] else 4
    if any(not t for t in tokens):raise ValueError("incomplete vocabulary")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix(".gguf.part")
    w=gguf.GGUFWriter(tmp,"openthai_systemone",use_temp_file=True)
    w.add_name("OpenThai-SystemOne v0.3")
    w.add_string("general.source.url",f"https://huggingface.co/{REPO}")
    w.add_string("general.source.revision",REVISION)
    w.add_string("general.license","apache-2.0")
    w.add_string("openthai.config",json.dumps(cfg,separators=(",",":")))
    w.add_uint32("openthai.context_length",a.context)
    w.add_uint32("openthai.state_length",min(32768,a.context//2))
    w.add_string("tokenizer.ggml.model","gpt2")
    w.add_string("tokenizer.ggml.pre","qwen3")
    w.add_string("tokenizer.normalizer","NFC")
    w.add_token_list(tokens);w.add_token_types(types)
    w.add_token_merges([" ".join(m) if isinstance(m,list) else m for m in tok["model"]["merges"]])
    w.add_string("tokenizer.pretokenizer",json.dumps(tok["pre_tokenizer"],ensure_ascii=False))
    count=0;max_loss=0.
    # NumPy does not expose BF16 directly; the raw uint16 bits shift exactly to F32.
    import struct
    with (a.source/"model.safetensors").open("rb") as f:
        length=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(length));base=8+length
        for name,t in header.items():
            if name=="__metadata__":continue
            lo,hi=t["data_offsets"];f.seek(base+lo)
            dtype=t["dtype"]
            raw=np.frombuffer(f.read(hi-lo),dtype={"BF16":"<u2","F32":"<f4","F16":"<f2"}[dtype])
            data=((raw.astype(np.uint32)<<16).view(np.float32) if dtype=="BF16" else raw.astype(np.float32)).reshape(t["shape"])
            if not np.isfinite(data).all():raise ValueError(f"nonfinite tensor {name}")
            out=data.astype(np.float16) if data.ndim>=2 else data
            if not np.isfinite(out).all():raise ValueError(f"F16 overflow {name}")
            max_loss=max(max_loss,float(np.max(np.abs(data-out.astype(np.float32)))))
            w.add_tensor(name,out)
            del raw,data,out
            count+=1
    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
    tmp.replace(a.output)
    with a.output.open("rb") as f:
        sha=hashlib.file_digest(f,"sha256").hexdigest()
    report=dict(source=REPO,revision=REVISION,output=str(a.output),tensor_count=count,
                max_abs_weight_conversion_error=max_loss,sha256=sha,context_length=a.context,
                format="GGUF v3, openthai_systemone architecture, F16 matrices/F32 vectors")
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
