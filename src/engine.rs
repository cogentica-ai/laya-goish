//! A loaded backend and its tokenizer, shared by CLI and HTTP.
use crate::{
    gguf::Gguf,
    json::J,
    models::{self, Backend, ModelInfo},
    request,
    tokenizer::Tokenizer,
};
use alloc::{boxed::Box, string::String};
pub struct Engine {
    pub info: ModelInfo,
    backend: Box<dyn Backend>,
    tokenizer: Tokenizer,
}
impl Engine {
    pub fn new(g: Gguf, threads: usize) -> Result<Self, String> {
        crate::tensor::check_cpu()?;
        let info = models::describe(&g)?;
        let tokenizer = Tokenizer::new(&g)?;
        let backend = models::load(g, threads)?;
        Ok(Self {
            info,
            backend,
            tokenizer,
        })
    }
    pub fn name(&self) -> &str {
        &self.info.name
    }
    pub fn layers(&self) -> usize {
        self.backend.layers()
    }
    pub fn validate(&self, req: &J) -> Result<(), String> {
        self.backend.validate(&self.tokenizer, req)
    }
    pub fn decide(&self, req: &J, raw: bool, canceled: &dyn Fn() -> bool) -> Result<J, String> {
        let resolved = request::resolve(req)?;
        self.backend
            .decide(&self.tokenizer, &resolved, raw, canceled)
    }
}
