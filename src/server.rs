use crate::{
    engine::Engine,
    io,
    json::{self, J},
    registry::Registry,
    request,
};
use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use goish::{
    go,
    net::{self, http},
    nil, os, time,
};

type Writer = dyn http::ResponseWriter + Send + Sync + 'static;
const MAX_BODY: i64 = 1024 * 1024;
struct App {
    registry: Registry,
    presets: J,
    key: String,
    raw: bool,
    busy: AtomicBool,
    draining: AtomicBool,
    requests: AtomicU64,
}
struct Busy<'a>(&'a AtomicBool);
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
fn reply(w: &Writer, code: i64, body: J) {
    w.Header()
        .Set("Content-Type", "application/json; charset=utf-8");
    w.Header().Set("Cache-Control", "no-store");
    w.Header().Set("X-Content-Type-Options", "nosniff");
    w.WriteHeader(code);
    w.Write(goish::bytes(goish::string::from(
        (body.dump() + "\n").as_str(),
    )));
}
fn error(w: &Writer, code: i64, kind: &str, message: &str) {
    reply(
        w,
        code,
        J::object(vec![(
            "error",
            J::object(vec![("type", J::s(kind)), ("message", J::s(message))]),
        )]),
    );
}
fn secret_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |v, (a, b)| v | (a ^ b)) == 0
}
impl App {
    fn validate(&self, model: &Engine, req: &J) -> Result<J, String> {
        if !matches!(req, J::Obj(_)) {
            return Err("request body must be a JSON object".into());
        }
        if req.get("state") == &J::Null && req.get("preset").str().is_empty() {
            return Err("state is required (or supply a preset)".into());
        }
        let resolved = request::resolve(req)?;
        model.validate(&resolved)?;
        Ok(resolved)
    }
    fn handle(&self, w: &Writer, r: &http::Request) {
        let request_id = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        w.Header()
            .Set("X-Request-Id", format!("laya-{request_id}").as_str());
        let path = r.URL.Path.to_string();
        let method = r.Method.to_string();
        let allowed = match path.as_str() {
            "/" | "/health" | "/v1/models" | "/v1/presets" => "GET, HEAD",
            "/v1/systemone" | "/v1/decide" | "/v1/decide/batch" => "POST",
            _ => {
                error(w, 404, "not_found_error", "Endpoint not found");
                return;
            }
        };
        if (method != "GET" && method != "HEAD" && allowed == "GET, HEAD")
            || (method != "POST" && allowed == "POST")
        {
            w.Header().Set("Allow", allowed);
            error(w, 405, "invalid_request_error", "Method not allowed");
            return;
        }
        let public = path == "/" || path == "/health" || path == "/v1/presets";
        if !public && !self.key.is_empty() {
            let auth = r.Header.Get("Authorization").to_string();
            if !auth
                .strip_prefix("Bearer ")
                .map(|s| secret_equal(s.as_bytes(), self.key.as_bytes()))
                .unwrap_or(false)
            {
                w.Header().Set("WWW-Authenticate", "Bearer");
                error(
                    w,
                    401,
                    "authentication_error",
                    "Invalid or missing bearer token",
                );
                return;
            }
        }
        match path.as_str() {
            "/" => {
                reply(
                    w,
                    200,
                    J::object(vec![
                        ("service", J::s("Laya / Goish Rust")),
                        ("health", J::s("/health")),
                        ("decide", J::s("/v1/systemone")),
                        ("presets", J::s("/v1/presets")),
                    ]),
                );
                return;
            }
            "/health" => {
                let draining = self.draining.load(Ordering::Acquire);
                reply(
                    w,
                    if draining { 503 } else { 200 },
                    J::object(vec![
                        ("status", J::s(if draining { "draining" } else { "ok" })),
                        ("model", J::s(self.registry.default_model().name())),
                        (
                            "family",
                            J::s(self.registry.default_model().info.family.as_str()),
                        ),
                        ("device", J::s("cpu")),
                        ("models", self.registry.loaded_names()),
                        ("busy", J::Bool(self.busy.load(Ordering::Acquire))),
                    ]),
                );
                return;
            }
            "/v1/models" => {
                reply(w, 200, self.registry.listing());
                return;
            }
            "/v1/presets" => {
                reply(
                    w,
                    200,
                    J::Arr(
                        self.presets
                            .obj()
                            .iter()
                            .map(|(n, p)| {
                                let mut fields = p.obj().to_vec();
                                fields.insert(0, ("name".into(), J::s(n.clone())));
                                J::Obj(fields)
                            })
                            .collect(),
                    ),
                );
                return;
            }
            _ => {}
        }
        if self.draining.load(Ordering::Acquire) {
            error(w, 503, "unavailable_error", "Server is shutting down");
            return;
        }
        let content_type = r.Header.Get("Content-Type").to_string();
        if !content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
        {
            error(
                w,
                415,
                "invalid_request_error",
                "Content-Type must be application/json",
            );
            return;
        }
        if r.ContentLength > MAX_BODY {
            error(
                w,
                413,
                "invalid_request_error",
                "Request body exceeds 1 MiB",
            );
            return;
        }
        // Goish alpha.13 buffers request bodies before dispatch (16 MiB hard cap).
        // Enforce our smaller API limit for both fixed-length and chunked input.
        let mut reader = http::MaxBytesReader(None, r.Body.clone(), MAX_BODY);
        let (body, err) = goish::io::ReadAll(&mut reader);
        if err != nil {
            error(
                w,
                413,
                "invalid_request_error",
                "Request body exceeds 1 MiB or could not be read",
            );
            return;
        }
        let text = match core::str::from_utf8(body.as_ref()) {
            Ok(v) => v,
            Err(_) => {
                error(w, 400, "invalid_request_error", "Request body is not UTF-8");
                return;
            }
        };
        let req = match json::parse(text) {
            Ok(v) => v,
            Err(e) => {
                error(w, 400, "invalid_request_error", &e);
                return;
            }
        };
        let model = match self.registry.select(&req) {
            Ok(model) => model,
            Err(e) => {
                error(w, 422, "invalid_request_error", &e);
                return;
            }
        };
        let batch = path == "/v1/decide/batch";
        let mut requests = Vec::new();
        if batch {
            if req.get("states").arr().is_empty() || req.get("states").arr().len() > 256 {
                error(
                    w,
                    422,
                    "invalid_request_error",
                    "states must be an array of 1..256 states",
                );
                return;
            }
            for state in req.get("states").arr() {
                requests.push(J::object(vec![
                    ("state", state.clone()),
                    ("questions", req.get("questions").clone()),
                    ("model", req.get("model").clone()),
                    ("order_invariant", req.get("order_invariant").clone()),
                    ("permutations", req.get("permutations").clone()),
                ]));
            }
        } else {
            requests.push(req);
        }
        let mut resolved = Vec::new();
        let mut question_count = 0;
        // Validate the whole batch before starting any expensive work.
        for req in &requests {
            match self.validate(model, req) {
                Ok(v) => {
                    question_count += v.get("questions").obj().len();
                    resolved.push(v);
                }
                Err(e) => {
                    error(w, 422, "invalid_request_error", &e);
                    return;
                }
            }
        }
        if question_count > 256 {
            error(
                w,
                422,
                "invalid_request_error",
                "At most 256 total questions per request",
            );
            return;
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            w.Header().Set("Retry-After", "1");
            error(
                w,
                429,
                "busy_error",
                "An inference request is running; retry later",
            );
            return;
        }
        let _busy = Busy(&self.busy);
        let start = time::Now();
        let mut results = Vec::new();
        let ctx = r.Context();
        for req in &resolved {
            match model.decide(req, self.raw, &|| ctx.Err() != nil) {
                Ok(v) => results.push(v),
                Err(e) => {
                    let canceled = ctx.Err() != nil;
                    error(
                        w,
                        if canceled { 408 } else { 500 },
                        if canceled {
                            "request_canceled"
                        } else {
                            "internal_error"
                        },
                        &e,
                    );
                    return;
                }
            }
        }
        w.Header().Set(
            "X-Response-Time-Ms",
            format!("{}", time::Since(start).Milliseconds()).as_str(),
        );
        reply(
            w,
            200,
            if batch {
                J::object(vec![("results", J::Arr(results))])
            } else {
                results.remove(0)
            },
        );
    }
}
pub fn serve(registry: Registry, host: &str, port: u16, raw: bool) -> Result<(), String> {
    let mut key = os::Getenv("LAYA_API_KEY").to_string();
    if key.is_empty() {
        key = os::Getenv("TYPESAFE_API_KEY").to_string();
    }
    let auth_enabled = !key.is_empty();
    let app = Arc::new(App {
        registry,
        presets: json::parse(include_str!("../presets.json"))?,
        key,
        raw,
        busy: AtomicBool::new(false),
        draining: AtomicBool::new(false),
        requests: AtomicU64::new(0),
    });
    let handler_app = app.clone();
    let server = Arc::new(http::Server {
        Handler: Arc::new(http::HandlerFunc(move |w: &Writer, r: &http::Request| {
            handler_app.handle(w, r)
        })),
        ReadHeaderTimeout: time::Second * 5,
        ReadTimeout: time::Second * 15,
        IdleTimeout: time::Second * 30,
        MaxHeaderBytes: 16 * 1024,
        MaxConcurrentConns: 32,
        ..Default::default()
    });
    let addr = net::JoinHostPort(
        goish::string::from(host),
        goish::string::from(port.to_string().as_str()),
    );
    let (listener, err) = net::Listen("tcp", addr.clone());
    if err != nil {
        return Err(format!("listen {addr}: {err:?}"));
    }
    let (ctx, stop) = os::signal::NotifyContext(
        goish::context::Background(),
        &[goish::syscall::SIGINT, goish::syscall::SIGTERM],
    );
    let done = goish::make!(chan bool,1);
    let shutdown_done = done.clone();
    let shutdown = server.clone();
    go!(move || {
        ctx.Done().Recv();
        app.draining.store(true, Ordering::Release);
        io::log("laya HTTP: draining requests (30 second deadline)");
        let err = shutdown.clone().Shutdown(time::Second * 30);
        if err != nil {
            shutdown.Close();
        }
        shutdown_done.Send(err == nil);
    });
    io::log(&format!(
        "laya HTTP listening at http://{addr} (bearer auth: {})",
        if auth_enabled { "on" } else { "off" }
    ));
    let err = server.Serve(listener);
    let closed = goish::errors::Is(err.clone(), http::ErrServerClosed);
    if closed {
        let (clean, _) = done.Recv();
        stop();
        if !clean {
            return Err("HTTP shutdown deadline exceeded".into());
        }
        return Ok(());
    }
    stop();
    Err(format!("HTTP server failed: {err:?}"))
}
