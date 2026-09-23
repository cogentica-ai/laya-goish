//! Qwen3.5 text tower plus OpenThai's slot head (Apache-2.0 upstream equations).
//! F32 activations; recurrent state is local to each forward, never shared across requests.
use crate::{
    gguf::{self, Gguf},
    json::{self, J},
    tensor::{self, Matrix},
};
use alloc::{string::String, vec, vec::Vec};

struct Full {
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    qn: Vec<f32>,
    kn: Vec<f32>,
}
struct Delta {
    qkv: Matrix,
    z: Matrix,
    a: Matrix,
    b: Matrix,
    o: Matrix,
    conv: Vec<f32>,
    dt: Vec<f32>,
    alog: Vec<f32>,
    norm: Vec<f32>,
}
enum Mixer {
    Full(Full),
    Delta(Delta),
}
struct Layer {
    n1: Vec<f32>,
    n2: Vec<f32>,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    mix: Mixer,
}
pub struct Model {
    pub g: Gguf,
    pub layers: usize,
    pub max_tokens: usize,
    pub max_state: usize,
    threads: usize,
    blocks: Vec<Layer>,
    norm: Vec<f32>,
    head: Matrix,
    bias: Vec<f32>,
    pub temperature: Vec<f32>,
}
fn matrix(g: &Gguf, key: &str, rows: usize, cols: usize) -> Result<Matrix, String> {
    let (t, _) = g.data(key)?;
    if t.dims != [cols, rows] {
        return Err(format!("{key}: unexpected matrix shape"));
    }
    Ok(Matrix {
        cols,
        rows,
        data: g.floats(key)?,
    })
}
fn vector(g: &Gguf, key: &str, n: usize) -> Result<Vec<f32>, String> {
    let (t, _) = g.data(key)?;
    if t.dims.iter().product::<usize>() != n {
        return Err(format!("{key}: unexpected vector shape"));
    }
    g.floats(key)
}
fn sigmoid(x: f32) -> f32 {
    1. / (1. + libm::expf(-x))
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        libm::log1pf(libm::expf(x))
    }
}
fn rms(x: &[f32], w: &[f32], offset: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = vec![0.; x.len()];
    for (x, y) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let scale = 1. / libm::sqrtf(tensor::dot(x, x) / d as f32 + 1e-6);
        for j in 0..d {
            y[j] = x[j] * scale * (offset + w[j]);
        }
    }
    out
}
impl Model {
    pub fn new(g: Gguf, threads: usize) -> Result<Self, String> {
        let cfg = json::parse(g.get("openthai.config").str())?;
        let c = cfg.get("text_config");
        for (k, n) in [
            ("hidden_size", 1024),
            ("intermediate_size", 3584),
            ("num_hidden_layers", 24),
            ("head_dim", 256),
            ("num_attention_heads", 8),
            ("num_key_value_heads", 2),
            ("linear_conv_kernel_dim", 4),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 16),
            ("vocab_size", 248339),
        ] {
            if c.get(k).num() != n as f64 {
                return Err(format!("unsupported OpenThai config {k}"));
            }
        }
        if cfg.get("n_slots").num() != 256.
            || cfg.get("abstain_slot").num() != 255.
            || cfg.get("answer_token_id").num() != 248082.
            || c.get("rms_norm_eps").num() != 1e-6
            || c.get("rope_parameters").get("rope_theta").num() != 10000000.
            || c.get("rope_parameters").get("partial_rotary_factor").num() != 0.25
            || c.get("rope_parameters").get("rope_type").str() != "default"
        {
            return Err("unsupported OpenThai head/normalization/rotary configuration".into());
        }
        let (emb, _) = g.data("model.embed_tokens.weight")?;
        if emb.dims != [1024, 248339] || ![0, 1].contains(&emb.kind) {
            return Err("OpenThai embeddings must be F32/F16 with shape [248339,1024]".into());
        }
        let max_tokens = g.num("openthai.context_length", 4096);
        let max_state = g.num("openthai.state_length", 2048);
        if !(32..=8192).contains(&max_tokens) || max_state < 1 || max_state > max_tokens {
            return Err("invalid OpenThai sequence budgets".into());
        }
        let mut blocks = Vec::new();
        for i in 0..24 {
            let p = format!("model.layers.{i}");
            let m = |tail: &str, r, c| matrix(&g, &format!("{p}.{tail}.weight"), r, c);
            let v = |tail: &str, n| vector(&g, &format!("{p}.{tail}"), n);
            let mix = match c.get("layer_types").arr().get(i).unwrap_or(&J::Null).str() {
                "full_attention" => Mixer::Full(Full {
                    q: m("self_attn.q_proj", 4096, 1024)?,
                    k: m("self_attn.k_proj", 512, 1024)?,
                    v: m("self_attn.v_proj", 512, 1024)?,
                    o: m("self_attn.o_proj", 1024, 2048)?,
                    qn: v("self_attn.q_norm.weight", 256)?,
                    kn: v("self_attn.k_norm.weight", 256)?,
                }),
                "linear_attention" => Mixer::Delta(Delta {
                    qkv: m("linear_attn.in_proj_qkv", 6144, 1024)?,
                    z: m("linear_attn.in_proj_z", 2048, 1024)?,
                    a: m("linear_attn.in_proj_a", 16, 1024)?,
                    b: m("linear_attn.in_proj_b", 16, 1024)?,
                    o: m("linear_attn.out_proj", 1024, 2048)?,
                    conv: v("linear_attn.conv1d.weight", 6144 * 4)?,
                    dt: v("linear_attn.dt_bias", 16)?,
                    alog: v("linear_attn.A_log", 16)?,
                    norm: v("linear_attn.norm.weight", 128)?,
                }),
                _ => return Err("unsupported OpenThai layer type".into()),
            };
            blocks.push(Layer {
                n1: v("input_layernorm.weight", 1024)?,
                n2: v("post_attention_layernorm.weight", 1024)?,
                gate: m("mlp.gate_proj", 3584, 1024)?,
                up: m("mlp.up_proj", 3584, 1024)?,
                down: m("mlp.down_proj", 1024, 3584)?,
                mix,
            });
        }
        let norm = vector(&g, "model.norm.weight", 1024)?;
        let head = matrix(&g, "slot_head.weight", 256, 1024)?;
        let bias = vector(&g, "slot_head.bias", 256)?;
        let temperature = vector(&g, "log_temperature", 3)?
            .iter()
            .map(|v| libm::expf(*v))
            .collect();
        Ok(Self {
            g,
            layers: 24,
            max_tokens,
            max_state,
            threads,
            blocks,
            norm,
            head,
            bias,
            temperature,
        })
    }
    pub fn forward(
        &self,
        ids: &[usize],
        positions: &[usize],
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty()
            || ids.len() > self.max_tokens
            || ids.iter().any(|&i| i >= 248339)
            || positions.iter().any(|&p| p >= ids.len())
        {
            return Err("invalid OpenThai token sequence".into());
        }
        let (emb, bytes) = self.g.data("model.embed_tokens.weight")?;
        let size = if emb.kind == 0 { 4 } else { 2 };
        let mut x = Vec::with_capacity(ids.len() * 1024);
        for &id in ids {
            x.extend(gguf::decode(
                emb.kind,
                &bytes[id * 1024 * size..(id + 1) * 1024 * size],
            )?);
        }
        for b in &self.blocks {
            if canceled() {
                return Err("request canceled".into());
            }
            let h = rms(&x, &b.n1, 1.);
            let y = match &b.mix {
                Mixer::Full(f) => f.run(&h, self.threads),
                Mixer::Delta(d) => d.run(&h, self.threads),
            };
            tensor::add(&mut x, &y);
            let h = rms(&x, &b.n2, 1.);
            let mut y = b.gate.apply(&h, None, self.threads);
            let up = b.up.apply(&h, None, self.threads);
            for (v, u) in y.iter_mut().zip(up) {
                *v = silu(*v) * u;
            }
            tensor::add(&mut x, &b.down.apply(&y, None, self.threads));
        }
        let mut answers = Vec::with_capacity(positions.len() * 1024);
        for &p in positions {
            answers.extend_from_slice(&x[p * 1024..(p + 1) * 1024]);
        }
        Ok(self.head.apply(
            &rms(&answers, &self.norm, 1.),
            Some(&self.bias),
            self.threads,
        ))
    }
}
impl Full {
    fn run(&self, x: &[f32], threads: usize) -> Vec<f32> {
        let s = x.len() / 1024;
        let qgate = self.q.apply(x, None, threads);
        let mut q = vec![0.; s * 2048];
        let mut gate = q.clone();
        for i in 0..s {
            for h in 0..8 {
                q[i * 2048 + h * 256..i * 2048 + (h + 1) * 256]
                    .copy_from_slice(&qgate[i * 4096 + h * 512..i * 4096 + h * 512 + 256]);
                gate[i * 2048 + h * 256..i * 2048 + (h + 1) * 256]
                    .copy_from_slice(&qgate[i * 4096 + h * 512 + 256..i * 4096 + (h + 1) * 512]);
            }
        }
        q = rms(&q, &self.qn, 1.);
        let mut k = rms(&self.k.apply(x, None, threads), &self.kn, 1.);
        let v = self.v.apply(x, None, threads);
        for (buf, heads) in [(&mut q, 8), (&mut k, 2)] {
            for i in 0..s {
                for h in 0..heads {
                    for j in 0..32 {
                        let angle = i as f32 / libm::powf(10000000., (2 * j) as f32 / 64.);
                        let co = libm::cosf(angle);
                        let si = libm::sinf(angle);
                        let p = i * heads * 256 + h * 256 + j;
                        let a = buf[p];
                        let b = buf[p + 32];
                        buf[p] = a * co - b * si;
                        buf[p + 32] = b * co + a * si;
                    }
                }
            }
        }
        let mut out = vec![0.; s * 2048];
        let mut scores = vec![0.; s];
        for i in 0..s {
            for h in 0..8 {
                let kh = h / 4;
                for j in 0..=i {
                    scores[j] = tensor::dot(
                        &q[i * 2048 + h * 256..][..256],
                        &k[j * 512 + kh * 256..][..256],
                    ) / 16.;
                }
                let p = tensor::softmax(&scores[..=i]);
                for (j, p) in p.iter().enumerate() {
                    for d in 0..256 {
                        out[i * 2048 + h * 256 + d] += p * v[j * 512 + kh * 256 + d];
                    }
                }
            }
        }
        for (o, g) in out.iter_mut().zip(gate) {
            *o *= sigmoid(g);
        }
        self.o.apply(&out, None, threads)
    }
}
impl Delta {
    fn run(&self, x: &[f32], threads: usize) -> Vec<f32> {
        let s = x.len() / 1024;
        let raw = self.qkv.apply(x, None, threads);
        let z = self.z.apply(x, None, threads);
        let a = self.a.apply(x, None, threads);
        let b = self.b.apply(x, None, threads);
        let mut qkv = vec![0.; raw.len()];
        for i in 0..s {
            for c in 0..6144 {
                let mut v = 0.;
                for j in 0..4 {
                    if i + j >= 3 {
                        v += raw[(i + j - 3) * 6144 + c] * self.conv[c * 4 + j];
                    }
                }
                qkv[i * 6144 + c] = silu(v);
            }
        }
        let mut out = vec![0.; s * 2048];
        // State is [key_dim, value_dim], maintained independently for each head.
        for h in 0..16 {
            let mut state = vec![0.; 128 * 128];
            for i in 0..s {
                let q = &qkv[i * 6144 + h * 128..][..128];
                let k = &qkv[i * 6144 + 2048 + h * 128..][..128];
                let v = &qkv[i * 6144 + 4096 + h * 128..][..128];
                let qs = 1. / libm::sqrtf(tensor::dot(q, q) + 1e-6) / libm::sqrtf(128.);
                let ks = 1. / libm::sqrtf(tensor::dot(k, k) + 1e-6);
                let decay =
                    libm::expf(-libm::expf(self.alog[h]) * softplus(a[i * 16 + h] + self.dt[h]));
                let beta = sigmoid(b[i * 16 + h]);
                let mut delta = [0.; 128];
                for v in state.iter_mut() {
                    *v *= decay;
                }
                for j in 0..128 {
                    let key = k[j] * ks;
                    for d in 0..128 {
                        delta[d] += state[j * 128 + d] * key;
                    }
                }
                for d in 0..128 {
                    delta[d] = (v[d] - delta[d]) * beta;
                }
                let o = &mut out[i * 2048 + h * 128..][..128];
                for j in 0..128 {
                    let key = k[j] * ks;
                    let query = q[j] * qs;
                    for d in 0..128 {
                        state[j * 128 + d] += key * delta[d];
                        o[d] += state[j * 128 + d] * query;
                    }
                }
            }
        }
        let mut out = rms(&out, &self.norm, 0.);
        for (o, z) in out.iter_mut().zip(z) {
            *o *= silu(z);
        }
        self.o.apply(&out, None, threads)
    }
}
