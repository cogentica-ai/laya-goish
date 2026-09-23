//! OpenThai-SystemOne Qwen3.5 text tower and slot decision head.
pub mod decision;
pub mod model;
use crate::{json::J, models::Backend, tokenizer::Tokenizer};
use alloc::string::String;
impl Backend for model::Model {
    fn layers(&self) -> usize {
        self.layers
    }
    fn validate(&self, t: &Tokenizer, req: &J) -> Result<(), String> {
        decision::validate(self, t, req)
    }
    fn decide(
        &self,
        t: &Tokenizer,
        req: &J,
        raw: bool,
        canceled: &dyn Fn() -> bool,
    ) -> Result<J, String> {
        decision::decide(self, t, req, raw, canceled)
    }
}
