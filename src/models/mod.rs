//! Backend contract and GGUF architecture dispatch. Add model-specific code in a subdirectory.
pub mod laya;
pub mod openthai;
use crate::{gguf::Gguf, json::J, tokenizer::Tokenizer};
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Backends own immutable weights; request-local state must not escape a call.
pub trait Backend: Send + Sync {
    fn layers(&self) -> usize;
    fn validate(&self, tokenizer: &Tokenizer, req: &J) -> Result<(), String>;
    fn decide(
        &self,
        tokenizer: &Tokenizer,
        req: &J,
        raw: bool,
        canceled: &dyn Fn() -> bool,
    ) -> Result<J, String>;
}
pub struct ModelInfo {
    pub name: String,
    pub family: String,
    pub architecture: String,
    pub aliases: Vec<String>,
    pub release_date: &'static str,
}
/// Inspect routing identity before allocating/dequantizing any model weights.
pub fn describe(g: &Gguf) -> Result<ModelInfo, String> {
    let architecture = g.get("general.architecture").str();
    match architecture {
        "openthai_systemone" => Ok(ModelInfo {
            name: "openthai-systemone".into(),
            family: "openthai-systemone".into(),
            architecture: architecture.into(),
            aliases: Vec::new(),
            release_date: "2026-09-22",
        }),
        "ggmlc" => {
            let name = g.get("laya.model_name").str();
            let family = match name {
                "laya" => "english",
                "laya-multilingual" => "multilingual",
                "laya-typed-decisions" => "typed-decisions",
                _ => return Err(format!("unsupported ggmlc model {name}")),
            };
            let mut aliases = vec![family.to_string()];
            if family == "english" {
                aliases.push("jev-latest".into());
            }
            Ok(ModelInfo {
                name: name.into(),
                family: family.into(),
                architecture: architecture.into(),
                aliases,
                release_date: "2026-09-20",
            })
        }
        _ => Err(format!("unsupported GGUF architecture {architecture}")),
    }
}
pub fn load(g: Gguf, threads: usize) -> Result<Box<dyn Backend>, String> {
    match g.get("general.architecture").str() {
        "ggmlc" => Ok(Box::new(laya::model::Model::new(g, threads)?)),
        "openthai_systemone" => Ok(Box::new(openthai::model::Model::new(g, threads)?)),
        name => Err(format!("unsupported GGUF architecture {name}")),
    }
}
