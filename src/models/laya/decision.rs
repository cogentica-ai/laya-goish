use super::model::Model;
use crate::{
    json::{self, J},
    tensor,
    tokenizer::Tokenizer,
};
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
pub struct Question {
    pub id: String,
    pub kind: usize,
    pub ins: String,
    pub labels: Vec<String>,
    pub opts: Vec<String>,
}
pub fn questions(req: &J) -> Result<Vec<Question>, String> {
    let obj = req.get("questions");
    if obj.obj().is_empty() || obj.obj().len() > 256 {
        return Err("questions must contain 1..256 named questions".into());
    }
    let mut out = Vec::new();
    for (id, q) in obj.obj() {
        let kind = match q.get("type").str() {
            "choice" => 0,
            "score" => 1,
            "noul" => 2,
            _ => return Err(format!("{id}: type must be choice, score, or noul")),
        };
        let ins = match q.get("instructions") {
            J::Str(s) => s.clone(),
            J::Null => String::new(),
            _ => return Err(format!("{id}: instructions must be a string")),
        };
        let mut labels = Vec::new();
        let mut opts = Vec::new();
        let c = q.get("criteria");
        let text = |v: &J| {
            if let J::Str(s) = v {
                s.clone()
            } else if *v == J::Null {
                String::new()
            } else {
                v.state()
            }
        };
        match kind {
            0 => {
                if c.obj().is_empty() {
                    return Err(format!("{id}: choice requires a criteria object"));
                }
                for (k, v) in c.obj() {
                    labels.push(k.clone());
                    let desc = text(v);
                    opts.push(if desc.is_empty() {
                        k.clone()
                    } else {
                        format!("{k}: {desc}")
                    })
                }
            }
            1 => {
                if c.arr().len() < 2 {
                    return Err(format!(
                        "{id}: score requires at least two ordered criteria"
                    ));
                }
                for (i, v) in c.arr().iter().enumerate() {
                    labels.push(text(v));
                    opts.push(format!("level {i}: {}", text(v)))
                }
            }
            _ => {
                for (k, default) in [
                    ("false", "no, the statement does not hold"),
                    ("true", "yes, the statement holds"),
                ] {
                    let v = text(c.get(k));
                    labels.push(k.to_string());
                    opts.push(format!("{k}: {}", if v.is_empty() { default } else { &v }))
                }
            }
        }
        if opts.is_empty() || opts.len() > 16 {
            return Err(format!("{id}: requires 1..16 options"));
        }
        out.push(Question {
            id: id.clone(),
            kind,
            ins,
            labels,
            opts,
        });
    }
    Ok(out)
}
pub fn encode(
    m: &Model,
    t: &Tokenizer,
    state: &str,
    q: &Question,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let max = m.g.num("laya.max_len", 512);
    let head_max = m.g.num("laya.head_max_len", 192);
    if head_max < 32 || head_max >= max {
        return Err("invalid sequence budgets".into());
    }
    let mask = m.g.num("laya.mask_token_id", 50284);
    let sep = m.g.num("laya.sep_token_id", 50282);
    let cls = m.g.num("laya.cls_token_id", 50281);
    let mut head = t.encode(&format!(
        "{} question: {}",
        ["choice", "score", "noul"][q.kind],
        q.ins.replace(&t.mask, " ")
    ))?;
    let mut opts = Vec::new();
    for o in &q.opts {
        let mut v = vec![mask];
        let mut tok = t.encode(&format!(" {}", o.replace(&t.mask, " ")))?;
        tok.truncate(48);
        v.extend(tok);
        opts.push(v)
    }
    let mut sum: usize = opts.iter().map(|v| v.len()).sum();
    if (head_max as isize) - (sum as isize) < 16 {
        let per = ((head_max - 16) / opts.len()).max(4);
        for o in &mut opts {
            o.truncate(per)
        }
        sum = opts.iter().map(|v| v.len()).sum();
    }
    head.truncate(head_max.saturating_sub(sum).max(8));
    let mut ids = vec![cls];
    ids.extend(head);
    ids.push(sep);
    let mut markers = Vec::new();
    for o in opts {
        markers.push(ids.len());
        ids.extend(o)
    }
    ids.push(sep);
    let room = max.saturating_sub(ids.len() + 1);
    let mut st = t.encode(&state.replace(&t.mask, " "))?;
    st.truncate(room);
    ids.extend(st);
    ids.push(sep);
    ids.truncate(max);
    if markers.iter().any(|&p| p >= ids.len()) {
        return Err("option markers exceed context".into());
    }
    Ok((ids, markers))
}
pub fn confidence(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.;
    }
    let e = -p
        .iter()
        .map(|&v| {
            let x = (v as f64).max(1e-12);
            x * libm::log(x)
        })
        .sum::<f64>();
    (1. - e / libm::log(p.len() as f64)).clamp(0., 1.) as f32
}
pub fn decide_with_cancel(
    m: &Model,
    t: &Tokenizer,
    req: &J,
    raw: bool,
    canceled: &dyn Fn() -> bool,
) -> Result<J, String> {
    let qs = questions(req)?;
    let state = match req.get("state") {
        J::Str(s) => s.clone(),
        J::Null => String::new(),
        v => v.state(),
    };
    let temp = json::parse(m.g.get("laya.temperature").str()).unwrap_or(J::Null);
    let buckets = json::parse(m.g.get("laya.temperature_by_options").str()).unwrap_or(J::Null);
    let start = goish::time::Now();
    let mut tokens = 0;
    let mut answers = Vec::new();
    for q in &qs {
        if canceled() {
            return Err("request canceled".into());
        }
        let (ids, markers) = encode(m, t, &state, q)?;
        tokens += ids.len();
        let (logits, act) = m.forward(&ids, &markers, q.kind)?;
        let k = logits.len();
        let bucket = format!(
            "{}:{}",
            ["choice", "score", "noul"][q.kind],
            if k <= 2 {
                "2"
            } else if k <= 5 {
                "3-5"
            } else if k <= 10 {
                "6-10"
            } else {
                "11+"
            }
        );
        let tv = buckets.get(&bucket);
        let temp = if *tv != J::Null {
            tv.num() as f32
        } else {
            temp.arr().get(q.kind).map(|v| v.num() as f32).unwrap_or(1.)
        }
        .max(1e-3);
        let p = tensor::softmax(&logits.iter().map(|v| v / temp).collect::<Vec<_>>());
        let ap = tensor::softmax(&act)[0];
        let mut a = vec![(
            "type".to_string(),
            J::s(["choice", "score", "noul"][q.kind]),
        )];
        let mut probs = Vec::new();
        for (i, &v) in p.iter().enumerate() {
            probs.push((
                if q.kind == 1 {
                    i.to_string()
                } else {
                    q.labels[i].clone()
                },
                J::n(v as f64),
            ))
        }
        match q.kind {
            0 => {
                let best = (0..k)
                    .max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap().then_with(|| b.cmp(&a)))
                    .unwrap();
                a.push(("choice".into(), J::s(q.labels[best].clone())))
            }
            1 => {
                a.push((
                    "score".into(),
                    J::n(
                        p.iter()
                            .enumerate()
                            .map(|(i, p)| i as f64 * (*p as f64))
                            .sum::<f64>(),
                    ),
                ));
                a.push((
                    "legend".into(),
                    J::Obj(
                        q.labels
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (i.to_string(), J::s(s.clone())))
                            .collect(),
                    ),
                ))
            }
            _ => a.push(("noul".into(), J::n(p[1] as f64))),
        }
        a.push((
            "confidence".into(),
            J::n(if q.kind == 2 {
                p[1].max(1. - p[1]) as f64
            } else {
                confidence(&p) as f64
            }),
        ));
        a.push(("probabilities".into(), J::Obj(probs)));
        a.push((
            "action".into(),
            J::object(vec![("act_probability", J::n(ap as f64))]),
        ));
        if raw {
            a.push((
                "logits".into(),
                J::Arr(logits.iter().map(|&v| J::n(v as f64)).collect()),
            ));
            a.push((
                "act_logits".into(),
                J::Arr(act.iter().map(|&v| J::n(v as f64)).collect()),
            ));
            a.push((
                "input_ids".into(),
                J::Arr(ids.iter().map(|&v| J::n(v as f64)).collect()),
            ));
            a.push((
                "markers".into(),
                J::Arr(markers.iter().map(|&v| J::n(v as f64)).collect()),
            ))
        }
        answers.push((q.id.clone(), J::Obj(a)));
    }
    let elapsed = goish::time::Since(start);
    let mut out = J::object(vec![
        ("model", m.g.get("laya.model_name").clone()),
        ("answers", J::Obj(answers)),
        (
            "usage",
            J::object(vec![
                ("input_tokens", J::n(tokens as f64)),
                ("output_tokens", J::n(0.)),
                ("latency_ms", J::n(elapsed.Milliseconds() as f64)),
            ]),
        ),
    ]);
    if let J::Obj(v) = &mut out {
        if *req.get("id") != J::Null {
            v.push(("id".into(), req.get("id").clone()))
        }
    }
    Ok(out)
}
pub fn self_test() -> Result<(), String> {
    if J::n(1.23456789).state() != "1.23457"
        || J::n(0.000000123456789).state() != "1.23457e-07"
        || J::n(1e20).state() != "1e+20"
    {
        return Err("state number serialization mismatch".into());
    }
    let j = json::parse(
        r#"{"state":"test","questions":{"z":{"type":"choice","criteria":{"zebra":null,"apple":"fruit"}},"a":{"type":"noul"}}}"#,
    )?;
    let q = questions(&j)?;
    if q[0].id != "z" || q[0].labels[0] != "zebra" {
        return Err("criteria order lost".into());
    }
    if json::parse(r#"{"x":1,"x":2}"#).is_ok() || json::parse("{} trailing").is_ok() {
        return Err("malformed JSON accepted".into());
    }
    if confidence(&[0.5, 0.5]).abs() > 1e-6 || confidence(&[1., 0.]) < 0.9999 {
        return Err("confidence mismatch".into());
    }
    Ok(())
}
