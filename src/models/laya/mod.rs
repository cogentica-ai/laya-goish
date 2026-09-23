//! Laya ModernBERT encoder and typed decision head.
pub mod decision;
pub mod model;
use crate::{json::J, models::Backend, tokenizer::Tokenizer};
use alloc::string::String;
impl Backend for model::Model {
    fn layers(&self) -> usize {
        self.layers
    }
    fn validate(&self, _: &Tokenizer, req: &J) -> Result<(), String> {
        decision::questions(req)?;
        Ok(())
    }
    fn decide(
        &self,
        t: &Tokenizer,
        req: &J,
        raw: bool,
        canceled: &dyn Fn() -> bool,
    ) -> Result<J, String> {
        decision::decide_with_cancel(self, t, req, raw, canceled)
    }
}
