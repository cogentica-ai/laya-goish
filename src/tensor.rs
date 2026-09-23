use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::arch::x86_64::*;
use goish::{go, sync::WaitGroup};
pub struct Matrix {
    pub cols: usize,
    pub rows: usize,
    pub data: Vec<f32>,
}
impl Matrix {
    pub fn apply(&self, x: &[f32], bias: Option<&[f32]>, threads: usize) -> Vec<f32> {
        assert_eq!(x.len() % self.cols, 0);
        let m = x.len() / self.cols;
        let n = self.rows;
        let k = self.cols;
        assert!(bias.map(|b| b.len() == n).unwrap_or(true));
        let mut out = vec![0.; m * n];
        let count = threads.min(m.div_ceil(4)).max(1);
        let per = m.div_ceil(count);
        let xp = x.as_ptr() as usize;
        let wp = self.data.as_ptr() as usize;
        let op = out.as_mut_ptr() as usize;
        let bp = bias.map(|b| b.as_ptr() as usize).unwrap_or(0);
        let wg = Arc::new(WaitGroup::new());
        wg.Add((count - 1) as i64);
        // Each worker owns disjoint output token rows. Inputs live through Wait().
        // Kernels cannot panic after these validated dimensions; no pointer escapes.
        for t in 1..count {
            let wg = wg.clone();
            let start = t * per;
            let end = ((t + 1) * per).min(m);
            go!(move || {
                run_kernel(xp, wp, op, bp, k, n, start, end);
                wg.Done();
            });
        }
        run_kernel(xp, wp, op, bp, k, n, 0, per.min(m));
        wg.Wait();
        out
    }
}
// Goish alpha.13 async preemption uses FXSAVE (XMM only), not XSAVE.
// Keep YMM state within a non-preemptible, allocation-free leaf. The kernel
// returns (including vzeroupper) before releasing the M; scheduling remains
// enabled between matrix operations. Do not move AVX into this wrapper.
#[inline(never)]
fn run_kernel(
    xp: usize,
    wp: usize,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    start: usize,
    end: usize,
) {
    goish::runtime::sched::acquirem();
    unsafe {
        kernel(xp, wp, op, bp, k, n, start, end);
    }
    goish::runtime::sched::releasem();
}
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel(
    xp: usize,
    wp: usize,
    op: usize,
    bp: usize,
    k: usize,
    n: usize,
    start: usize,
    end: usize,
) {
    let x = xp as *const f32;
    let w = wp as *const f32;
    let o = op as *mut f32;
    let bias = bp as *const f32;
    let mut r = start;
    while r < end {
        let nr = (end - r).min(4);
        for c in 0..n {
            let mut a = [_mm256_setzero_ps(); 4];
            let mut j = 0;
            while j + 8 <= k {
                let v = _mm256_loadu_ps(w.add(c * k + j));
                for t in 0..nr {
                    a[t] = _mm256_fmadd_ps(v, _mm256_loadu_ps(x.add((r + t) * k + j)), a[t])
                }
                j += 8
            }
            for t in 0..nr {
                let mut lanes = [0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), a[t]);
                let mut v = lanes.iter().sum::<f32>();
                for q in j..k {
                    v += *w.add(c * k + q) * *x.add((r + t) * k + q)
                }
                if bp != 0 {
                    v += *bias.add(c)
                }
                *o.add((r + t) * n + c) = v;
            }
        }
        r += nr;
    }
}
pub fn norm(x: &[f32], d: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut y = vec![0.; x.len()];
    for (a, o) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let mean = a.iter().sum::<f32>() / d as f32;
        let var = a
            .iter()
            .map(|v| {
                let z = v - mean;
                z * z
            })
            .sum::<f32>()
            / d as f32;
        let scale = 1. / libm::sqrtf(var + eps);
        for j in 0..d {
            o[j] = (a[j] - mean) * scale * w.map(|w| w[j]).unwrap_or(1.)
                + b.map(|b| b[j]).unwrap_or(0.)
        }
    }
    y
}
pub fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a += b
    }
}
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1. + libm::tanhf(0.7978845608028654 * x * (1. + 0.044715 * x * x)))
}
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<_> = x.iter().map(|v| libm::expf(v - m)).collect();
    let sum = p.iter().sum::<f32>();
    for v in &mut p {
        *v /= sum
    }
    p
}
pub fn attention(
    qkv: &[f32],
    s: usize,
    d: usize,
    hd: usize,
    rope: Option<(&[f32], &[f32])>,
    window: Option<usize>,
) -> Vec<f32> {
    let heads = d / hd;
    let mut q = vec![0.; s * d];
    let mut k = q.clone();
    let mut v = q.clone();
    for i in 0..s {
        q[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d..i * 3 * d + d]);
        k[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
        v[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + 2 * d..(i + 1) * 3 * d]);
    }
    if let Some((cos, sin)) = rope {
        for a in [&mut q, &mut k] {
            for i in 0..s {
                for h in 0..heads {
                    for j in 0..hd / 2 {
                        let p = i * d + h * hd + j;
                        let u = a[p];
                        let v = a[p + hd / 2];
                        a[p] = u * cos[i * hd + j] - v * sin[i * hd + j];
                        a[p + hd / 2] = v * cos[i * hd + j + hd / 2] + u * sin[i * hd + j + hd / 2];
                    }
                }
            }
        }
    }
    let mut out = vec![0.; s * d];
    let scale = 1. / libm::sqrtf(hd as f32);
    let mut score = vec![0.; s];
    for h in 0..heads {
        for i in 0..s {
            let lo = window.map(|w| i.saturating_sub(w)).unwrap_or(0);
            let hi = window.map(|w| (i + w + 1).min(s)).unwrap_or(s);
            for j in lo..hi {
                score[j] = dot(&q[i * d + h * hd..][..hd], &k[j * d + h * hd..][..hd]) * scale
            }
            let p = softmax(&score[lo..hi]);
            for (j, &p) in (lo..hi).zip(&p) {
                for z in 0..hd {
                    out[i * d + h * hd + z] += p * v[j * d + h * hd + z]
                }
            }
        }
    }
    out
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn self_test() -> Result<(), String> {
    let w = Matrix {
        cols: 17,
        rows: 11,
        data: (0..187).map(|i| (i as f32 - 83.) / 71.).collect(),
    };
    let x: Vec<f32> = (0..153).map(|i| (i as f32 - 71.) / 43.).collect();
    let b = vec![0.25; 11];
    for threads in [1, 2, 4] {
        let y = w.apply(&x, Some(&b), threads);
        for r in 0..9 {
            for c in 0..11 {
                let want = dot(&x[r * 17..][..17], &w.data[c * 17..][..17]) + 0.25;
                if (y[r * 11 + c] - want).abs() > 0.0001 {
                    return Err("matrix kernel mismatch".into());
                }
            }
        }
    }
    if (crate::gguf::half(0x3c00) - 1.).abs() > 1e-8 || crate::gguf::half(1) != 1. / 16777216. {
        return Err("half conversion mismatch".into());
    }
    Ok(())
}
