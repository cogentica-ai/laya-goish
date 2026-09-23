use crate::{
    gguf::Gguf,
    json::J,
    tensor::{self, Matrix},
};
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
struct Linear {
    w: Matrix,
    b: Option<Vec<f32>>,
}
impl Linear {
    fn run(&self, x: &[f32], threads: usize) -> Vec<f32> {
        self.w.apply(x, self.b.as_deref(), threads)
    }
}
struct Norm {
    w: Option<Vec<f32>>,
    b: Option<Vec<f32>>,
    eps: f32,
}
impl Norm {
    fn run(&self, x: &[f32], d: usize) -> Vec<f32> {
        tensor::norm(x, d, self.w.as_deref(), self.b.as_deref(), self.eps)
    }
}
struct Layer {
    n1: Option<Norm>,
    qkv: Linear,
    wo: Linear,
    n2: Norm,
    wi: Linear,
    down: Linear,
    full: bool,
}
pub struct Model {
    pub g: Gguf,
    pub layers: usize,
    pub d: usize,
    hd: usize,
    threads: usize,
    emb: Matrix,
    embnorm: Norm,
    finalnorm: Norm,
    trunk: Vec<Layer>,
    head: Vec<Layer>,
    type_emb: Vec<f32>,
    scorenorm: Norm,
    score1: Linear,
    score2: Linear,
    act1: Linear,
    act2: Linear,
    cos_full: Vec<f32>,
    sin_full: Vec<f32>,
    cos_slide: Vec<f32>,
    sin_slide: Vec<f32>,
}
fn name<'a>(graph: &'a J, id: &J) -> &'a str {
    graph
        .get("tensors")
        .get(&format!("{}", id.num() as usize))
        .get("name")
        .str()
}
fn linear(g: &Gguf, graph: &J, key: &str) -> Result<Linear, String> {
    let weight = format!("{key}.weight");
    let (t, _) = g.data(&weight)?;
    let cols = t.dims[0];
    let rows = *t.dims.get(1).unwrap_or(&1);
    if t.dims.iter().skip(2).any(|&d| d != 1) {
        return Err("non-matrix linear".into());
    }
    let node = graph
        .get("nodes")
        .arr()
        .iter()
        .find(|n| {
            n.get("opcode").num() == 29.
                && n.get("inputs")
                    .arr()
                    .first()
                    .map(|i| name(graph, i) == weight)
                    .unwrap_or(false)
        })
        .ok_or_else(|| format!("graph lacks linear {key}"))?;
    let mut biases = Vec::new();
    if node.get("inputs").arr().len() > 2 {
        biases.push(name(graph, &node.get("inputs").arr()[2]).to_string())
    }
    for n in graph.get("nodes").arr() {
        if n.get("opcode").num() == 200.
            && n.get("inputs").arr().first() == node.get("outputs").arr().first()
        {
            biases.push(name(graph, &n.get("inputs").arr()[1]).to_string())
        }
    }
    let b = if biases.is_empty() {
        None
    } else {
        let mut out = vec![0.; rows];
        for name in biases {
            let v = g.floats(&name)?;
            if v.len() != rows {
                return Err("linear bias shape mismatch".into());
            }
            tensor::add(&mut out, &v)
        }
        Some(out)
    };
    Ok(Linear {
        w: Matrix {
            cols,
            rows,
            data: g.floats(&weight)?,
        },
        b,
    })
}
fn norm(g: &Gguf, graph: &J, i: usize) -> Result<Norm, String> {
    let key = if i == 0 {
        "layer_norm".to_string()
    } else {
        format!("layer_norm_{i}")
    };
    let node = graph
        .get("nodes")
        .arr()
        .iter()
        .find(|n| n.get("name").str() == key && n.get("opcode").num() == 201.)
        .ok_or_else(|| format!("missing {key}"))?;
    let ins = node.get("inputs").arr();
    let get = |idx: usize| -> Result<Option<Vec<f32>>, String> {
        if idx < ins.len() {
            Ok(Some(g.floats(name(graph, &ins[idx]))?))
        } else {
            Ok(None)
        }
    };
    let eps = node.get("attributes").get("eps").num() as f32;
    if !(eps > 0. && eps < 0.1) {
        return Err("invalid layer norm epsilon".into());
    }
    Ok(Norm {
        w: get(1)?,
        b: get(2)?,
        eps,
    })
}
impl Model {
    pub fn new(g: Gguf, threads: usize) -> Result<Self, String> {
        // These kernels require AVX2/FMA; fail explicitly on other CPUs.
        let f = core::arch::x86_64::__cpuid(1);
        let f7 = core::arch::x86_64::__cpuid_count(7, 0);
        if f.ecx & (1 << 12) == 0 || f.ecx & (1 << 27) == 0 || f7.ebx & (1 << 5) == 0 {
            return Err("CPU requires AVX2, FMA and OSXSAVE".into());
        }
        if unsafe { core::arch::x86_64::_xgetbv(0) } & 6 != 6 {
            return Err("OS has not enabled AVX state".into());
        }
        if g.get("general.architecture").str() != "ggmlc"
            || !g.get("laya.checkpoint").str().contains("laya")
        {
            return Err(
                "expected a compiled Laya GGUF (Kev and other architectures are outside this port)"
                    .into(),
            );
        }
        if g.num("laya.max_opts", 16) != 16 {
            return Err("expected 16-option Laya head".into());
        }
        if !g.get("ggmlc.decision").str().is_empty() {
            let r = crate::json::parse(g.get("ggmlc.decision").str())?;
            if r.get("kind").str() != "laya" {
                return Err("unsupported decision sequence recipe".into());
            }
        }
        let graph = g.graph()?;
        let mut layers = 0;
        while g.tensors.contains_key(&format!("qkv.{layers}.weight")) {
            layers += 1
        }
        let mut heads = 0;
        while g.tensors.contains_key(&format!("head_qkv.{heads}.weight")) {
            heads += 1
        }
        if layers == 0 || layers > 64 || heads != 2 {
            return Err("unsupported Laya encoder/head layout".into());
        }
        let et = g.tensors.get("tok_emb.weight").ok_or("missing embedding")?;
        let d = et.dims[0];
        let vocab = et.dims[1];
        let hd = g.tensors.get("cos_full").ok_or("missing RoPE tables")?.dims[0];
        if d == 0 || hd == 0 || hd % 2 != 0 || d % hd != 0 {
            return Err("invalid attention dimensions".into());
        }
        let attns: Vec<_> = graph
            .get("nodes")
            .arr()
            .iter()
            .filter(|n| n.get("opcode").num() == 74.)
            .collect();
        if attns.len() != layers + heads {
            return Err("unexpected attention graph".into());
        }
        let fullmask = attns[0]
            .get("inputs")
            .arr()
            .last()
            .ok_or("attention has no mask")?;
        let mut trunk = Vec::new();
        for i in 0..layers {
            trunk.push(Layer {
                n1: if i == 0 {
                    None
                } else {
                    Some(norm(&g, &graph, 2 * i)?)
                },
                qkv: linear(&g, &graph, &format!("qkv.{i}"))?,
                wo: linear(&g, &graph, &format!("wo.{i}"))?,
                n2: norm(&g, &graph, 2 * i + 1)?,
                wi: linear(&g, &graph, &format!("mlp_wi.{i}"))?,
                down: linear(&g, &graph, &format!("mlp_wo.{i}"))?,
                full: attns[i].get("inputs").arr().last() == Some(fullmask),
            });
        }
        let mut head = Vec::new();
        for i in 0..heads {
            head.push(Layer {
                n1: Some(norm(&g, &graph, 2 * layers + 1 + 2 * i)?),
                qkv: linear(&g, &graph, &format!("head_qkv.{i}"))?,
                wo: linear(&g, &graph, &format!("head_wo.{i}"))?,
                n2: norm(&g, &graph, 2 * layers + 2 + 2 * i)?,
                wi: linear(&g, &graph, &format!("head_fc1.{i}"))?,
                down: linear(&g, &graph, &format!("head_fc2.{i}"))?,
                full: true,
            });
        }
        for (l, ishead) in trunk
            .iter()
            .map(|l| (l, false))
            .chain(head.iter().map(|l| (l, true)))
        {
            if l.qkv.w.cols != d
                || l.qkv.w.rows != 3 * d
                || l.wo.w.cols != d
                || l.wo.w.rows != d
                || l.wi.w.cols != d
                || l.down.w.rows != d
                || l.wi.w.rows != l.down.w.cols * if ishead { 1 } else { 2 }
            {
                return Err("inconsistent layer dimensions".into());
            }
            for n in l.n1.iter().chain(core::iter::once(&l.n2)) {
                if n.w.as_ref().map(|w| w.len() != d).unwrap_or(false)
                    || n.b.as_ref().map(|b| b.len() != d).unwrap_or(false)
                {
                    return Err("norm shape mismatch".into());
                }
            }
        }
        let out = Self {
            layers,
            d,
            hd,
            threads,
            emb: Matrix {
                cols: d,
                rows: vocab,
                data: g.floats("tok_emb.weight")?,
            },
            embnorm: norm(&g, &graph, 0)?,
            finalnorm: norm(&g, &graph, 2 * layers)?,
            trunk,
            head,
            type_emb: g.floats("type_emb.weight")?,
            scorenorm: norm(&g, &graph, 2 * layers + 1 + 2 * heads)?,
            score1: linear(&g, &graph, "scorer_fc1")?,
            score2: linear(&g, &graph, "scorer_fc2")?,
            act1: linear(&g, &graph, "act_fc1")?,
            act2: linear(&g, &graph, "act_fc2")?,
            cos_full: g.floats("cos_full")?,
            sin_full: g.floats("sin_full")?,
            cos_slide: g.floats("cos_slide")?,
            sin_slide: g.floats("sin_slide")?,
            g,
        };
        if out.type_emb.len() != 3 * d
            || out.score1.w.cols != d
            || out.score1.w.rows != d
            || out.score2.w.cols != d
            || out.score2.w.rows != 1
            || out.act1.w.cols != d + 4
            || out.act2.w.cols != out.act1.w.rows
            || out.act2.w.rows != 2
        {
            return Err("decision head shape mismatch".into());
        }
        Ok(out)
    }
    pub fn forward(
        &self,
        ids: &[usize],
        markers: &[usize],
        qtype: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let s = ids.len();
        let d = self.d;
        let threads = self.threads;
        if s == 0
            || s > self.g.num("laya.max_len", 512)
            || qtype >= 3
            || markers.len() < 1
            || markers.len() > 16
            || markers.iter().any(|&p| p >= s)
        {
            return Err("invalid inference input shape".into());
        }
        if [
            &self.cos_full,
            &self.sin_full,
            &self.cos_slide,
            &self.sin_slide,
        ]
        .iter()
        .any(|a| a.len() < s * self.hd)
        {
            return Err("RoPE table too short".into());
        }
        let mut h = Vec::with_capacity(s * d);
        for &id in ids {
            if id >= self.emb.rows {
                return Err("token ID out of range".into());
            }
            h.extend_from_slice(&self.emb.data[id * d..(id + 1) * d])
        }
        h = self.embnorm.run(&h, d);
        for l in &self.trunk {
            let x = if let Some(n) = &l.n1 {
                n.run(&h, d)
            } else {
                h.clone()
            };
            let qkv = l.qkv.run(&x, threads);
            let (cos, sin) = if l.full {
                (&self.cos_full, &self.sin_full)
            } else {
                (&self.cos_slide, &self.sin_slide)
            };
            let a = tensor::attention(
                &qkv,
                s,
                d,
                self.hd,
                Some((cos, sin)),
                if l.full { None } else { Some(64) },
            );
            tensor::add(&mut h, &l.wo.run(&a, threads));
            let n = l.n2.run(&h, d);
            let up = l.wi.run(&n, threads);
            let f = l.down.w.cols;
            let mut m = vec![0.; s * f];
            for i in 0..s {
                for j in 0..f {
                    m[i * f + j] = tensor::gelu(up[i * 2 * f + j]) * up[i * 2 * f + f + j]
                }
            }
            tensor::add(&mut h, &l.down.run(&m, threads));
        }
        h = self.finalnorm.run(&h, d);
        for row in h.chunks_exact_mut(d) {
            tensor::add(row, &self.type_emb[qtype * d..(qtype + 1) * d]);
        }
        for l in &self.head {
            let x = l.n1.as_ref().unwrap().run(&h, d);
            let a = tensor::attention(&l.qkv.run(&x, threads), s, d, self.hd, None, None);
            tensor::add(&mut h, &l.wo.run(&a, threads));
            let x = l.n2.run(&h, d);
            let mut f = l.wi.run(&x, threads);
            for v in &mut f {
                *v = v.max(0.)
            }
            tensor::add(&mut h, &l.down.run(&f, threads));
        }
        let mut gathered = Vec::new();
        for &p in markers {
            gathered.extend_from_slice(&h[p * d..(p + 1) * d])
        }
        let x = self.scorenorm.run(&gathered, d);
        let mut x = self.score1.run(&x, threads);
        for v in &mut x {
            *v = tensor::gelu(*v)
        }
        let logits = self.score2.run(&x, threads);
        let p = tensor::softmax(&logits);
        let top1 = p.iter().copied().fold(0., f32::max);
        let top2 = p
            .iter()
            .map(|&p| p * (top1 - p) / ((top1 - p) + 1e-6))
            .fold(0., f32::max);
        let k = markers.len().max(2) as f32;
        let ent = -p.iter().map(|p| p * libm::logf(p.max(1e-9))).sum::<f32>() / libm::logf(k);
        let mut pooled = h[..d].to_vec();
        pooled.extend_from_slice(&[top1, top1 - top2, ent, k / 255.]);
        let mut x = self.act1.run(&pooled, threads);
        for v in &mut x {
            *v = tensor::gelu(*v)
        }
        let act = self.act2.run(&x, threads);
        if logits.iter().chain(&act).any(|x| !x.is_finite()) {
            return Err("non-finite model output".into());
        }
        Ok((logits, act))
    }
}
