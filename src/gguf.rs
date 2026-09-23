use crate::{
    io::Mapping,
    json::{self, J},
};
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec,
    vec::Vec,
};
pub struct Tensor {
    pub dims: Vec<usize>,
    pub kind: u32,
    pub offset: usize,
    pub len: usize,
}
pub struct Gguf {
    pub map: Mapping,
    pub meta: BTreeMap<String, J>,
    pub tensors: BTreeMap<String, Tensor>,
}
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("GGUF length overflow")?;
        let s = self.b.get(self.p..end).ok_or("truncated GGUF")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > 32 * 1024 * 1024 {
            return Err("GGUF string too large".into());
        }
        core::str::from_utf8(self.take(n)?)
            .map(|s| s.to_string())
            .map_err(|_| "invalid GGUF UTF-8".into())
    }
    fn value(&mut self, t: u32, depth: usize) -> Result<J, String> {
        if depth > 4 {
            return Err("GGUF metadata nesting too deep".into());
        }
        Ok(match t {
            0 => J::n(self.take(1)?[0] as f64),
            1 => J::n(self.take(1)?[0] as i8 as f64),
            2 => J::n(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as f64),
            3 => J::n(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as f64),
            4 => J::n(self.u32()? as f64),
            5 => J::n(self.u32()? as i32 as f64),
            6 => J::n(f32::from_bits(self.u32()?) as f64),
            7 => J::Bool(self.take(1)?[0] != 0),
            8 => J::s(self.string()?),
            9 => {
                let t = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1000000 {
                    return Err("GGUF array too large".into());
                }
                let mut a = Vec::with_capacity(n);
                for _ in 0..n {
                    a.push(self.value(t, depth + 1)?)
                }
                J::Arr(a)
            }
            10 => J::n(self.u64()? as f64),
            11 => J::n(self.u64()? as i64 as f64),
            12 => J::n(f64::from_bits(self.u64()?)),
            _ => return Err(format!("unsupported GGUF metadata type {t}")),
        })
    }
}
pub fn layout(kind: u32) -> Result<(usize, usize), String> {
    match kind {
        0 => Ok((1, 4)),
        1 => Ok((1, 2)),
        2 => Ok((32, 18)),
        8 => Ok((32, 34)),
        12 => Ok((256, 144)),
        14 => Ok((256, 210)),
        _ => Err(format!(
            "unsupported GGML tensor type {kind}; supported F32/F16/Q4_0/Q8_0/Q4_K/Q6_K"
        )),
    }
}
impl Gguf {
    pub fn open(path: &str) -> Result<Self, String> {
        let map = Mapping::open(path)?;
        let mut r = Reader {
            b: map.bytes(),
            p: 0,
        };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        if r.u32()? != 3 {
            return Err("expected GGUF version 3".into());
        }
        let nt = r.u64()?;
        let nk = r.u64()?;
        if nt > 100000 || nk > 100000 {
            return Err("GGUF table too large".into());
        }
        let mut meta = BTreeMap::new();
        for _ in 0..nk {
            let k = r.string()?;
            let t = r.u32()?;
            let v = r.value(t, 0)?;
            if meta.insert(k, v).is_some() {
                return Err("duplicate GGUF metadata key".into());
            }
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()?;
            if !(1..=4).contains(&nd) {
                return Err("invalid tensor rank".into());
            }
            let mut dims = Vec::new();
            let mut count = 1usize;
            for _ in 0..nd {
                let d = r.u64()? as usize;
                if d == 0 {
                    return Err("empty tensor".into());
                }
                count = count.checked_mul(d).ok_or("tensor size overflow")?;
                dims.push(d)
            }
            let kind = r.u32()?;
            let offset = r.u64()? as usize;
            let (block, size) = layout(kind)?;
            if dims[0] % block != 0 {
                return Err("invalid quantized row alignment".into());
            }
            let len = (count / block)
                .checked_mul(size)
                .ok_or("tensor byte size overflow")?;
            if tensors
                .insert(
                    name,
                    Tensor {
                        dims,
                        kind,
                        offset,
                        len,
                    },
                )
                .is_some()
            {
                return Err("duplicate tensor".into());
            }
        }
        let align = meta
            .get("general.alignment")
            .map(|v| v.num() as usize)
            .unwrap_or(32);
        if align == 0 || !align.is_power_of_two() || align > 4096 {
            return Err("invalid GGUF alignment".into());
        }
        let base = r.p.checked_add(align - 1).ok_or("offset overflow")? & !(align - 1);
        for t in tensors.values_mut() {
            if t.offset % align != 0 {
                return Err("unaligned GGUF tensor".into());
            }
            t.offset = t.offset.checked_add(base).ok_or("offset overflow")?;
            if t.offset.checked_add(t.len).ok_or("length overflow")? > map.bytes().len() {
                return Err("truncated tensor data".into());
            }
        }
        Ok(Self { map, meta, tensors })
    }
    pub fn get(&self, k: &str) -> &J {
        self.meta.get(k).unwrap_or(&J::Null)
    }
    pub fn num(&self, k: &str, default: usize) -> usize {
        self.meta
            .get(k)
            .map(|v| v.num() as usize)
            .unwrap_or(default)
    }
    pub fn graph(&self) -> Result<J, String> {
        json::parse(self.get("ggmlc.graph_spec").str())
    }
    pub fn data(&self, name: &str) -> Result<(&Tensor, &[u8]), String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        Ok((t, &self.map.bytes()[t.offset..t.offset + t.len]))
    }
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        let (t, b) = self.data(name)?;
        decode(t.kind, b)
    }
    pub fn info(&self) -> J {
        J::object(vec![
            ("name", self.get("general.name").clone()),
            (
                "model",
                if self.get("general.architecture").str() == "openthai_systemone" {
                    J::s("openthai-systemone")
                } else {
                    self.get("laya.model_name").clone()
                },
            ),
            ("architecture", self.get("general.architecture").clone()),
            ("checkpoint", self.get("laya.checkpoint").clone()),
            (
                "max_len",
                J::n(self.num("openthai.context_length", self.num("laya.max_len", 512)) as f64),
            ),
            ("tensors", J::n(self.tensors.len() as f64)),
            ("runtime", J::s("Goish Rust, no_std, CPU")),
        ])
    }
}
pub fn half(x: u16) -> f32 {
    let sign = ((x as u32) & 0x8000) << 16;
    let e = (x >> 10) & 31;
    let m = x & 1023;
    if e == 0 {
        if m == 0 {
            f32::from_bits(sign)
        } else {
            let v = (m as f32) * (1.0 / 16777216.0);
            if sign != 0 {
                -v
            } else {
                v
            }
        }
    } else if e == 31 {
        f32::from_bits(sign | 0x7f800000 | ((m as u32) << 13))
    } else {
        f32::from_bits(sign | (((e as u32) + 112) << 23) | ((m as u32) << 13))
    }
}
fn h(b: &[u8]) -> f32 {
    half(u16::from_le_bytes([b[0], b[1]]))
}
pub fn decode(kind: u32, b: &[u8]) -> Result<Vec<f32>, String> {
    let (block, size) = layout(kind)?;
    let mut out = vec![0.; b.len() / size * block];
    match kind {
        0 => {
            for (x, c) in out.iter_mut().zip(b.chunks_exact(4)) {
                *x = f32::from_le_bytes(c.try_into().unwrap())
            }
        }
        1 => {
            for (x, c) in out.iter_mut().zip(b.chunks_exact(2)) {
                *x = h(c)
            }
        }
        8 => {
            for (o, c) in out.chunks_exact_mut(32).zip(b.chunks_exact(34)) {
                let d = h(c);
                for j in 0..32 {
                    o[j] = d * (c[2 + j] as i8 as f32)
                }
            }
        }
        2 => {
            for (o, c) in out.chunks_exact_mut(32).zip(b.chunks_exact(18)) {
                let d = h(c);
                for j in 0..16 {
                    o[j] = d * ((c[2 + j] & 15) as f32 - 8.);
                    o[j + 16] = d * ((c[2 + j] >> 4) as f32 - 8.)
                }
            }
        }
        12 => {
            for (o, c) in out.chunks_exact_mut(256).zip(b.chunks_exact(144)) {
                let d = h(c);
                let dm = h(&c[2..]);
                let sc = &c[4..16];
                let qs = &c[16..];
                let scale = |j: usize| -> (u8, u8) {
                    if j < 4 {
                        (sc[j] & 63, sc[j + 4] & 63)
                    } else {
                        (
                            (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4),
                            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
                        )
                    }
                };
                for g in 0..4 {
                    let (s0, m0) = scale(g * 2);
                    let (s1, m1) = scale(g * 2 + 1);
                    for j in 0..32 {
                        let q = qs[g * 32 + j];
                        o[g * 64 + j] = d * s0 as f32 * (q & 15) as f32 - dm * m0 as f32;
                        o[g * 64 + 32 + j] = d * s1 as f32 * (q >> 4) as f32 - dm * m1 as f32
                    }
                }
            }
        }
        14 => {
            for (o, c) in out.chunks_exact_mut(256).zip(b.chunks_exact(210)) {
                let ql = &c[..128];
                let qh = &c[128..192];
                let sc = &c[192..208];
                let d = h(&c[208..]);
                for n in 0..2 {
                    for l in 0..32 {
                        let lo = n * 64 + l;
                        let hi = qh[n * 32 + l];
                        let si = n * 8 + l / 16;
                        let q = [
                            (ql[lo] & 15) | ((hi & 3) << 4),
                            (ql[lo + 32] & 15) | (((hi >> 2) & 3) << 4),
                            (ql[lo] >> 4) | (((hi >> 4) & 3) << 4),
                            (ql[lo + 32] >> 4) | (((hi >> 6) & 3) << 4),
                        ];
                        for j in 0..4 {
                            o[n * 128 + j * 32 + l] =
                                d * (sc[si + j * 2] as i8 as f32) * (q[j] as f32 - 32.)
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}
