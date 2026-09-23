use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}
impl J {
    pub fn get(&self, k: &str) -> &J {
        if let J::Obj(v) = self {
            if let Some((_, v)) = v.iter().find(|(n, _)| n == k) {
                return v;
            }
        }
        &J::Null
    }
    pub fn str(&self) -> &str {
        if let J::Str(s) = self {
            s
        } else {
            ""
        }
    }
    pub fn num(&self) -> f64 {
        if let J::Num(n) = self {
            *n
        } else {
            0.0
        }
    }
    pub fn arr(&self) -> &[J] {
        if let J::Arr(v) = self {
            v
        } else {
            &[]
        }
    }
    pub fn obj(&self) -> &[(String, J)] {
        if let J::Obj(v) = self {
            v
        } else {
            &[]
        }
    }
    pub fn s(s: impl Into<String>) -> J {
        J::Str(s.into())
    }
    pub fn n(n: impl Into<f64>) -> J {
        J::Num(n.into())
    }
    pub fn object(p: Vec<(&str, J)>) -> J {
        J::Obj(p.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn dump(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    // Match upstream's state serialization (spaces after separators, insertion order).
    pub fn state(&self) -> String {
        match self {
            J::Arr(a) => crate::format!(
                "[{}]",
                a.iter().map(|v| v.state()).collect::<Vec<_>>().join(", ")
            ),
            J::Obj(a) => crate::format!(
                "{{{}}}",
                a.iter()
                    .map(|(k, v)| crate::format!("{}: {}", J::s(k.as_str()).dump(), v.state()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            J::Num(n) if !(*n == (*n as i64) as f64 && n.abs() < 1e15) => {
                goish::strconv::FormatFloat(*n, b'g', 6, 64).to_string()
            }
            _ => self.dump(),
        }
    }
}
pub fn parse(s: &str) -> Result<J, String> {
    serde_json::from_str(s).map_err(|e| e.to_string())
}
impl Serialize for J {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            J::Null => s.serialize_unit(),
            J::Bool(b) => s.serialize_bool(*b),
            J::Num(n) => {
                if n.is_finite() && *n == (*n as i64) as f64 && n.abs() < 1e15 {
                    s.serialize_i64(*n as i64)
                } else {
                    s.serialize_f64(*n)
                }
            }
            J::Str(v) => s.serialize_str(v),
            J::Arr(v) => {
                let mut a = s.serialize_seq(Some(v.len()))?;
                for x in v {
                    a.serialize_element(x)?
                }
                a.end()
            }
            J::Obj(v) => {
                let mut o = s.serialize_map(Some(v.len()))?;
                for (k, x) in v {
                    o.serialize_entry(k, x)?
                }
                o.end()
            }
        }
    }
}
struct JV;
impl<'de> Visitor<'de> for JV {
    type Value = J;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON value")
    }
    fn visit_unit<E>(self) -> Result<J, E> {
        Ok(J::Null)
    }
    fn visit_bool<E>(self, v: bool) -> Result<J, E> {
        Ok(J::Bool(v))
    }
    fn visit_i64<E>(self, v: i64) -> Result<J, E> {
        Ok(J::Num(v as f64))
    }
    fn visit_u64<E>(self, v: u64) -> Result<J, E> {
        Ok(J::Num(v as f64))
    }
    fn visit_f64<E>(self, v: f64) -> Result<J, E> {
        Ok(J::Num(v))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<J, E> {
        Ok(J::s(v))
    }
    fn visit_string<E>(self, v: String) -> Result<J, E> {
        Ok(J::Str(v))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut v = Vec::new();
        while let Some(x) = a.next_element()? {
            v.push(x)
        }
        Ok(J::Arr(v))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<J, A::Error> {
        let mut v = Vec::new();
        while let Some((k, x)) = a.next_entry::<String, J>()? {
            if v.iter().any(|(old, _)| old == &k) {
                return Err(serde::de::Error::custom("duplicate JSON key"));
            }
            v.push((k, x))
        }
        Ok(J::Obj(v))
    }
}
impl<'de> Deserialize<'de> for J {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<J, D::Error> {
        d.deserialize_any(JV)
    }
}
