//! Shared preset expansion; backend-specific formatting lives with each model.
use crate::json::{self, J};
use alloc::{string::String, vec};
pub fn resolve(req: &J) -> Result<J, String> {
    let mut resolved = req.clone();
    if *req.get("questions") == J::Null && !req.get("preset").str().is_empty() {
        let presets = json::parse(include_str!("../presets.json"))?;
        let p = presets.get(req.get("preset").str());
        if *p == J::Null {
            return Err("unknown preset".into());
        }
        let mut state = if *req.get("state") == J::Null {
            p.get("state").clone()
        } else {
            req.get("state").clone()
        };
        if let J::Str(text) = req.get("text") {
            if let J::Obj(v) = &mut state {
                let key = p.get("state_key").str();
                if let Some((_, s)) = v.iter_mut().find(|(k, _)| k == key) {
                    *s = J::s(text.clone())
                } else {
                    v.push((key.into(), J::s(text.clone())));
                }
            } else {
                state = J::s(text.clone())
            }
        }
        resolved = J::object(vec![
            ("state", state),
            ("questions", p.get("questions").clone()),
            ("id", req.get("id").clone()),
            ("order_invariant", req.get("order_invariant").clone()),
            ("permutations", req.get("permutations").clone()),
        ]);
    }
    Ok(resolved)
}
