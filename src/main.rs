#![no_std]
#![no_main]
extern crate alloc;
#[macro_export]
macro_rules! format {
 ($($arg:tt)*) => {{
  let mut out = alloc::string::String::new();
  core::fmt::Write::write_fmt(&mut out, core::format_args!($($arg)*)).expect("string formatting");
  out
 }};
}
mod engine;
mod gguf;
mod io;
mod json;
mod models;
mod registry;
mod request;
mod server;
mod tensor;
mod tokenizer;
use alloc::{string::String, vec};
use json::J;
#[goish::main]
fn main() {
    if let Err(e) = run() {
        io::log(&format!("laya: {e}"));
        goish::os::Exit(1);
    }
}
fn run() -> Result<(), String> {
    let a = io::args();
    if a.len() < 2 || a[1] == "help" || a[1] == "--help" {
        io::out("Laya GGUF / Goish Rust\n  laya info MODEL\n  laya tokenize MODEL --text TEXT\n  laya decide MODEL --request FILE [--threads N] [--raw]\n  laya decide MODEL --preset email --text TEXT\n  laya list-presets\n  laya daemon MODEL [--threads N]\n  laya serve MODEL [--model MODEL ...] [--default-model NAME] [--host 127.0.0.1] [--port 8080] [--threads N]\n  laya self-test");
        return Ok(());
    }
    if a[1] == "list-presets" {
        let p = json::parse(include_str!("../presets.json"))?;
        io::out(&J::Arr(p.obj().iter().map(|(n, _)| J::s(n.clone())).collect()).dump());
        return Ok(());
    }
    if a[1] == "self-test" {
        tensor::self_test()?;
        models::laya::decision::self_test()?;
        io::out("self-test: PASS");
        return Ok(());
    }
    if !["info", "tokenize", "decide", "daemon", "serve"].contains(&a[1].as_str()) {
        return Err("unknown command; run laya help".into());
    }
    let path = a
        .get(2)
        .filter(|s| !s.starts_with("--"))
        .ok_or("missing model path")?;
    let mut i = 3;
    while i < a.len() {
        match a[i].as_str() {
            "--raw" | "--json" => i += 1,
            "--threads" | "--text" | "--request" | "--preset" | "--state" | "--state-file"
            | "--questions" | "--questions-file" | "--device" | "--host" | "--port" | "--model"
            | "--default-model" => {
                if i + 1 >= a.len() {
                    return Err(format!("missing value for {}", a[i]));
                }
                if (a[i] == "--model" || a[i] == "--default-model") && a[1] != "serve" {
                    return Err(format!("{} is only supported by serve", a[i]));
                }
                if a[i] == "--device" && a[i + 1] != "cpu" && a[i + 1] != "auto" {
                    return Err("this port supports CPU inference only".into());
                }
                i += 2;
            }
            _ => return Err(format!("unknown option {}", a[i])),
        }
    }
    let opt = |key: &str| a.iter().position(|s| s == key).and_then(|i| a.get(i + 1));
    let port = opt("--port")
        .map(|s| s.parse::<u16>())
        .transpose()
        .map_err(|_| "invalid --port")?
        .unwrap_or(8080);
    if port == 0 {
        return Err("--port must be 1..65535".into());
    }
    let host = opt("--host").map(|s| s.as_str()).unwrap_or("127.0.0.1");
    let threads = opt("--threads")
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid --threads")?
        .unwrap_or(4);
    if threads == 0 || threads > 64 {
        return Err("--threads must be 1..64".into());
    }
    goish::runtime::GOMAXPROCS(threads as i64);
    let raw = a.iter().any(|s| s == "--raw");
    if a[1] == "serve" {
        let mut paths = vec![path.clone()];
        let mut i = 3;
        while i < a.len() {
            if a[i] == "--model" {
                paths.push(a[i + 1].clone());
            }
            i += if a[i] == "--raw" || a[i] == "--json" {
                1
            } else {
                2
            };
        }
        let registry =
            registry::Registry::load(&paths, threads, opt("--default-model").map(|s| s.as_str()))?;
        return server::serve(registry, host, port, raw);
    }
    let g = gguf::Gguf::open(path)?;
    if a[1] == "info" {
        io::out(&g.info().dump());
        return Ok(());
    }
    if a[1] == "tokenize" {
        let tok = tokenizer::Tokenizer::new(&g)?;
        let text = opt("--text").ok_or("missing --text")?;
        io::out(
            &J::Arr(
                tok.encode(text)?
                    .into_iter()
                    .map(|n| J::n(n as f64))
                    .collect(),
            )
            .dump(),
        );
        return Ok(());
    }
    let m = engine::Engine::new(g, threads)?;
    io::log(&format!(
        "loaded {} ({} layers, {} threads)",
        path,
        m.layers(),
        threads
    ));
    if a[1] == "daemon" {
        while let Some(line) = io::line() {
            if line.trim().is_empty() {
                continue;
            }
            let r = json::parse(&line).and_then(|req| m.decide(&req, raw, &|| false));
            match r {
                Ok(j) => io::out(&j.dump()),
                Err(e) => io::out(&J::object(vec![("error", J::s(e))]).dump()),
            }
        }
        return Ok(());
    }
    if a[1] != "decide" {
        return Err("unknown command".into());
    }
    let req = if let Some(path) = opt("--request") {
        json::parse(&io::read(path)?)?
    } else {
        let presets = json::parse(include_str!("../presets.json"))?;
        let preset = presets.get(opt("--preset").map(|s| s.as_str()).unwrap_or("email"));
        if *preset == J::Null {
            return Err("unknown preset".into());
        }
        let mut state = if let Some(s) = opt("--state") {
            json::parse(s).unwrap_or(J::s(s.clone()))
        } else if let Some(p) = opt("--state-file") {
            json::parse(&io::read(p)?)?
        } else {
            preset.get("state").clone()
        };
        if let Some(text) = opt("--text") {
            if let J::Obj(v) = &mut state {
                let key = preset.get("state_key").str();
                if let Some((_, val)) = v.iter_mut().find(|(k, _)| k == key) {
                    *val = J::s(text.clone())
                } else {
                    v.push((key.into(), J::s(text.clone())));
                }
            } else {
                state = J::s(text.clone())
            }
        }
        let questions = if let Some(s) = opt("--questions") {
            json::parse(s)?
        } else if let Some(p) = opt("--questions-file") {
            json::parse(&io::read(p)?)?
        } else {
            preset.get("questions").clone()
        };
        J::object(vec![("state", state), ("questions", questions)])
    };
    io::out(&m.decide(&req, raw, &|| false)?.dump());
    Ok(())
}
