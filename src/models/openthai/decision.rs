//! OpenThai's causal control-token format and calibrated slot decoding.
use super::model::Model;
use crate::{json::J, tensor, tokenizer::Tokenizer};
use alloc::{
    collections::BTreeSet,
    string::{String, ToString},
    vec,
    vec::Vec,
};
pub struct Spec {
    id: String,
    kind: usize,
    ins: String,
    names: Vec<String>,
    desc: Vec<String>,
}
fn clean(s: &str) -> String {
    s.replace("<|ts_", "<\u{200b}|ts_")
}
pub fn questions(req: &J) -> Result<Vec<Spec>, String> {
    let qs = req.get("questions").obj();
    if qs.is_empty() || qs.len() > 256 {
        return Err("questions must contain 1..256 named questions".into());
    }
    if !matches!(req.get("state"), J::Str(_) | J::Obj(_) | J::Arr(_)) {
        return Err("OpenThai state must be a string, object, or array".into());
    }
    let mut out = Vec::new();
    for (id, q) in qs {
        let kind = match q.get("type").str() {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            _ => return Err(format!("{id}: invalid question type")),
        };
        let ins = match q.get("instructions") {
            J::Str(s) => s.clone(),
            _ => return Err(format!("{id}: instructions must be a string")),
        };
        let criteria = q.get("criteria");
        let mut names = Vec::new();
        let mut desc = Vec::new();
        let text = |v: &J| -> Result<String, String> {
            match v {
                J::Str(s) => Ok(s.clone()),
                J::Null => Ok(String::new()),
                _ => Err(format!("{id}: descriptions must be strings or null")),
            }
        };
        match kind {
            0 => {
                if criteria.obj().is_empty() || criteria.obj().len() > 255 {
                    return Err(format!("{id}: choice requires 1..255 options"));
                }
                for (name, d) in criteria.obj() {
                    if name.trim().is_empty() {
                        return Err(format!("{id}: option name is empty"));
                    }
                    names.push(name.clone());
                    desc.push(text(d)?);
                }
            }
            1 => {
                if !(2..=10).contains(&criteria.arr().len()) {
                    return Err(format!("{id}: score requires 2..10 levels"));
                }
                for (i, d) in criteria.arr().iter().enumerate() {
                    if !matches!(d, J::Str(_)) {
                        return Err(format!("{id}: score descriptions must be strings"));
                    }
                    names.push(i.to_string());
                    desc.push(text(d)?);
                }
            }
            _ => {
                if !matches!(criteria, J::Null | J::Obj(_))
                    || criteria
                        .obj()
                        .iter()
                        .any(|(k, _)| k != "false" && k != "true")
                {
                    return Err(format!("{id}: noul criteria keys must be false/true"));
                }
                names = vec!["no".into(), "yes".into()];
                desc = vec![text(criteria.get("false"))?, text(criteria.get("true"))?];
            }
        }
        out.push(Spec {
            id: id.clone(),
            kind,
            ins,
            names,
            desc,
        });
    }
    if !matches!(req.get("order_invariant"), J::Null | J::Bool(_)) {
        return Err("order_invariant must be boolean".into());
    }
    if let J::Num(n) = req.get("permutations") {
        if *n < 1. || *n > 32. || *n != (*n as usize) as f64 {
            return Err("permutations must be 1..32".into());
        }
    } else if req.get("permutations") != &J::Null {
        return Err("permutations must be 1..32".into());
    }
    Ok(out)
}
fn py_state(v: &J) -> String {
    match v {
        J::Str(s) => J::s(s.as_str()).dump(),
        J::Arr(a) => format!(
            "[{}]",
            a.iter().map(py_state).collect::<Vec<_>>().join(", ")
        ),
        J::Obj(a) => format!(
            "{{{}}}",
            a.iter()
                .map(|(k, v)| format!("{}: {}", J::s(k.as_str()).dump(), py_state(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.dump(),
    }
}
fn orders(k: usize, n: usize) -> Vec<Vec<usize>> {
    let n = n.min(k).max(1);
    let offsets: BTreeSet<_> = (0..n)
        .map(|j| {
            let a = j * k;
            let d = a / n;
            let rem = a % n;
            (d + usize::from(rem * 2 > n || rem * 2 == n && d % 2 == 1)) % k
        })
        .collect();
    offsets
        .into_iter()
        .map(|off| (0..k).map(|i| (i + off) % k).collect())
        .collect()
}
fn permutations(req: &J, qs: &[Spec]) -> usize {
    let k = qs
        .iter()
        .filter(|q| q.kind == 0)
        .map(|q| q.names.len())
        .max()
        .unwrap_or(1);
    let n = match req.get("permutations") {
        J::Num(n) => *n as usize,
        _ => match req.get("order_invariant") {
            J::Bool(false) => 1,
            J::Bool(true) => 8,
            _ => {
                if k >= 11 {
                    8
                } else {
                    1
                }
            }
        },
    };
    n.min(k).max(1)
}
pub struct Encoded {
    pub ids: Vec<usize>,
    pub positions: Vec<usize>,
    order: Vec<Vec<usize>>,
    pub truncated: bool,
}
fn encode(
    m: &Model,
    t: &Tokenizer,
    req: &J,
    qs: &[Spec],
    n: usize,
    pass: usize,
) -> Result<Encoded, String> {
    let mut blocks = Vec::new();
    let mut order = Vec::new();
    for q in qs {
        let ord = if q.kind == 0 {
            let all = orders(q.names.len(), n);
            all[pass % all.len()].clone()
        } else {
            (0..q.names.len()).collect()
        };
        let mut text = format!(
            "<|ts_q|><|ts_{}|> {}\n",
            ["choice", "score", "noul"][q.kind],
            clean(&q.ins).trim()
        );
        for (slot, &i) in ord.iter().enumerate() {
            text.push_str(&format!("<|ts_opt_{slot}|> {}", clean(&q.names[i]).trim()));
            if !q.desc[i].is_empty() {
                text.push_str(&format!(": {}", clean(&q.desc[i]).trim()));
            }
            text.push('\n');
        }
        text.push_str("<|ts_answer|>\n");
        blocks.push(t.encode(&text)?);
        order.push(ord);
    }
    let total: usize = blocks.iter().map(|b| b.len()).sum();
    if total >= m.max_tokens {
        return Err("question tokens exceed OpenThai context budget".into());
    }
    let state = match req.get("state") {
        J::Str(s) => s.clone(),
        v => py_state(v),
    };
    let mut ids = t.encode(&format!("<|ts_state|> {}\n", clean(&state).trim()))?;
    let budget = m.max_state.min(m.max_tokens - total);
    let truncated = ids.len() > budget;
    if truncated {
        let tail = ids[ids.len() - (budget - 1)..].to_vec();
        ids.truncate(1);
        ids.extend(tail);
    }
    let mut positions = Vec::new();
    for b in blocks {
        let p = b
            .iter()
            .rposition(|&id| id == 248082)
            .ok_or("missing OpenThai answer token")?;
        positions.push(ids.len() + p);
        ids.extend(b);
    }
    Ok(Encoded {
        ids,
        positions,
        order,
        truncated,
    })
}
pub fn validate(m: &Model, t: &Tokenizer, req: &J) -> Result<(), String> {
    let qs = questions(req)?;
    let n = permutations(req, &qs);
    // Validate every permutation before inference; ordering can change token counts.
    for pass in 0..n {
        encode(m, t, req, &qs, n, pass)?;
    }
    Ok(())
}
pub fn decide(
    m: &Model,
    t: &Tokenizer,
    req: &J,
    raw: bool,
    canceled: &dyn Fn() -> bool,
) -> Result<J, String> {
    let start = goish::time::Now();
    let qs = questions(req)?;
    let n = permutations(req, &qs);
    let mut probs: Vec<Vec<f32>> = qs.iter().map(|q| vec![0.; q.names.len()]).collect();
    let mut abstain = vec![0.; qs.len()];
    let mut tokens = 0;
    let mut truncated = false;
    let mut debug = Vec::new();
    for pass in 0..n {
        if canceled() {
            return Err("request canceled".into());
        }
        let e = encode(m, t, req, &qs, n, pass)?;
        if pass == 0 {
            tokens = e.ids.len();
        }
        truncated |= e.truncated;
        let logits = m.forward(&e.ids, &e.positions, canceled)?;
        for (i, q) in qs.iter().enumerate() {
            let k = q.names.len();
            let row = &logits[i * 256..(i + 1) * 256];
            let mut valid: Vec<f32> = row[..k].iter().map(|v| v / m.temperature[q.kind]).collect();
            valid.push(row[255] / m.temperature[q.kind]);
            let p = tensor::softmax(&valid);
            let sum = p[..k].iter().sum::<f32>().max(1e-12);
            for slot in 0..k {
                probs[i][e.order[i][slot]] += p[slot] / sum / n as f32;
            }
            abstain[i] += p[k] / n as f32;
        }
        if raw {
            debug.push(J::object(vec![
                (
                    "input_ids",
                    J::Arr(e.ids.iter().map(|&v| J::n(v as f64)).collect()),
                ),
                (
                    "answer_positions",
                    J::Arr(e.positions.iter().map(|&v| J::n(v as f64)).collect()),
                ),
                (
                    "logits",
                    J::Arr(logits.iter().map(|&v| J::n(v as f64)).collect()),
                ),
            ]));
        }
    }
    let mut answers = Vec::new();
    for (i, q) in qs.iter().enumerate() {
        let p = &mut probs[i];
        let sum = p.iter().sum::<f32>().max(1e-12);
        for v in p.iter_mut() {
            *v /= sum;
        }
        let k = p.len();
        let entropy = -p
            .iter()
            .map(|v| v.max(1e-12) * libm::logf(v.max(1e-12)))
            .sum::<f32>();
        let confidence = if k <= 1 {
            1.
        } else {
            (1. - entropy / libm::logf(k as f32)).clamp(0., 1.)
        };
        let mut fields = vec![("type", J::s(["choice", "score", "noul"][q.kind]))];
        if q.kind == 2 {
            fields.push(("noul", J::n(p[1] as f64)));
        } else {
            fields.push((
                "probabilities",
                J::Obj(
                    q.names
                        .iter()
                        .cloned()
                        .zip(p.iter().map(|&v| J::n(v as f64)))
                        .collect(),
                ),
            ));
            fields.push(("confidence", J::n(confidence as f64)));
            if q.kind == 0 {
                let mut best = 0;
                for j in 1..k {
                    if p[j] > p[best] {
                        best = j;
                    }
                }
                fields.push(("choice", J::s(q.names[best].clone())));
                fields.push(("abstain", J::n(abstain[i] as f64)));
            } else {
                fields.push((
                    "score",
                    J::n(p.iter().enumerate().map(|(j, v)| j as f32 * v).sum::<f32>() as f64),
                ));
                fields.push((
                    "legend",
                    J::Obj(
                        q.desc
                            .iter()
                            .enumerate()
                            .map(|(j, v)| {
                                (
                                    j.to_string(),
                                    J::s(if v.is_empty() {
                                        j.to_string()
                                    } else {
                                        v.clone()
                                    }),
                                )
                            })
                            .collect(),
                    ),
                ));
            }
        }
        answers.push((q.id.clone(), J::object(fields)));
    }
    let mut out = vec![
        ("model", J::s("openthai-systemone")),
        ("answers", J::Obj(answers)),
        (
            "usage",
            J::object(vec![
                ("input_tokens", J::n(tokens as f64)),
                ("output_tokens", J::n(0.)),
                ("permutations", J::n(n as f64)),
                ("truncated_state", J::Bool(truncated)),
                (
                    "latency_ms",
                    J::n(goish::time::Since(start).Milliseconds() as f64),
                ),
            ]),
        ),
    ];
    if req.get("id") != &J::Null {
        out.push(("id", req.get("id").clone()));
    }
    if raw {
        out.push(("_raw", J::Arr(debug)));
    }
    Ok(J::object(out))
}
