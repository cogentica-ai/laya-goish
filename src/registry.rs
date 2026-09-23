//! Immutable model registry. Routes aliases to one shared model instance.
use crate::{engine::Engine, gguf::Gguf, io, json::J, models};
use alloc::{collections::BTreeMap, string::String, vec, vec::Vec};

pub struct Registry {
    models: Vec<Engine>,
    names: BTreeMap<String, usize>,
    default: usize,
}
impl Registry {
    pub fn load(
        paths: &[String],
        threads: usize,
        default_name: Option<&str>,
    ) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("at least one model is required".into());
        }
        let mut pending = Vec::new();
        let mut names = BTreeMap::new();
        // Preflight every path, architecture, name and alias before expensive weight loading.
        for (index, path) in paths.iter().enumerate() {
            let g = Gguf::open(path)?;
            let info = models::describe(&g)?;
            for name in core::iter::once(&info.name).chain(info.aliases.iter()) {
                if names.insert(name.clone(), index).is_some() {
                    return Err(format!(
                        "duplicate model name or alias {name}; load one checkpoint per model"
                    ));
                }
            }
            pending.push(g);
        }
        let default = match default_name {
            Some(name) => *names
                .get(name)
                .ok_or_else(|| format!("unknown default model {name}"))?,
            None => 0,
        };
        let mut models = Vec::new();
        for (g, path) in pending.into_iter().zip(paths) {
            let model = Engine::new(g, threads)?;
            io::log(&format!(
                "loaded {} from {} ({} layers, {} threads)",
                model.name(),
                path,
                model.layers(),
                threads
            ));
            models.push(model);
        }
        Ok(Self {
            models,
            names,
            default,
        })
    }
    pub fn select(&self, req: &J) -> Result<&Engine, String> {
        if !matches!(req, J::Obj(_)) {
            return Err("request body must be a JSON object".into());
        }
        match req.get("model") {
            J::Null => Ok(self.default_model()),
            J::Str(name) => {
                let index = self
                    .names
                    .get(name)
                    .ok_or_else(|| format!("unknown model {name}; see /v1/models"))?;
                Ok(&self.models[*index])
            }
            _ => Err("model must be a string naming a loaded model".into()),
        }
    }
    pub fn default_model(&self) -> &Engine {
        &self.models[self.default]
    }
    pub fn loaded_names(&self) -> J {
        J::Arr(self.models.iter().map(|m| J::s(m.name())).collect())
    }
    pub fn listing(&self) -> J {
        let mut entries = Vec::new();
        for m in &self.models {
            for name in core::iter::once(&m.info.name).chain(m.info.aliases.iter()) {
                entries.push(J::object(vec![
                    ("name", J::s(name.as_str())),
                    ("canonical_name", J::s(m.name())),
                    ("architecture", J::s(m.info.architecture.as_str())),
                    ("family", J::s(m.info.family.as_str())),
                    ("default", J::Bool(m.name() == self.default_model().name())),
                    (
                        "description",
                        J::s(format!(
                            "Locally loaded {} decision model on Goish Rust",
                            m.name()
                        )),
                    ),
                    ("release_date", J::s(m.info.release_date)),
                ]));
            }
        }
        J::object(vec![
            ("models", J::Arr(entries)),
            ("default_model", J::s(self.default_model().name())),
        ])
    }
}
