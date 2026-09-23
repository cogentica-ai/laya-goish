use crate::gguf::Gguf;
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
pub struct Tokenizer {
    vocab: BTreeMap<String, usize>,
    ranks: BTreeMap<(String, String), usize>,
    bytes: Vec<String>,
    special: Vec<(String, usize)>,
    pub mask: String,
    gemma: bool,
    qwen: bool,
}
impl Tokenizer {
    pub fn new(g: &Gguf) -> Result<Self, String> {
        let pre = g.get("tokenizer.ggml.pre").str();
        if pre != "gpt2" && pre != "gemma" && pre != "qwen3" {
            return Err(format!("unsupported tokenizer {pre}"));
        }
        let tokens = g.get("tokenizer.ggml.tokens").arr();
        if tokens.is_empty() {
            return Err("missing vocabulary".into());
        }
        let vocab: BTreeMap<_, _> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.str().to_string(), i))
            .collect();
        let mut ranks = BTreeMap::new();
        for (i, m) in g.get("tokenizer.ggml.merges").arr().iter().enumerate() {
            if let Some((a, b)) = m.str().split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i);
            }
        }
        let mut extra = 256u32;
        let mut bytes = Vec::new();
        for i in 0..256 {
            let cp = if (33..=126).contains(&i)
                || (161..=172).contains(&i)
                || (174..=255).contains(&i)
            {
                i
            } else {
                let c = extra;
                extra += 1;
                c
            };
            bytes.push(char::from_u32(cp).unwrap().to_string())
        }
        let mut special: Vec<_> = vocab
            .iter()
            .filter(|(s, id)| {
                if pre == "qwen3" {
                    return g
                        .get("tokenizer.ggml.token_type")
                        .arr()
                        .get(**id)
                        .map(|t| t.num() == 3. || t.num() == 4.)
                        .unwrap_or(false);
                }
                s.len() >= 3
                    && ((s.starts_with('<') && s.ends_with('>'))
                        || (s.starts_with('[') && s.ends_with(']')))
            })
            .map(|(s, i)| (s.clone(), *i))
            .collect();
        special.sort_by_key(|(s, _)| core::cmp::Reverse(s.len()));
        let mask = tokens
            .get(g.num("laya.mask_token_id", 50284))
            .ok_or("mask ID outside vocabulary")?
            .str()
            .to_string();
        Ok(Self {
            vocab,
            ranks,
            bytes,
            special,
            mask,
            gemma: pre == "gemma",
            qwen: pre == "qwen3",
        })
    }
    fn bpe(&self, mut word: Vec<String>, out: &mut Vec<usize>) -> Result<(), String> {
        while word.len() > 1 {
            let best = (0..word.len() - 1)
                .filter_map(|i| {
                    self.ranks
                        .get(&(word[i].clone(), word[i + 1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            if let Some((_, i)) = best {
                let right = word.remove(i + 1);
                word[i].push_str(&right)
            } else {
                break;
            }
        }
        for p in word {
            if let Some(id) = self.vocab.get(&p) {
                out.push(*id)
            } else if self.gemma {
                for b in p.bytes() {
                    out.push(
                        *self
                            .vocab
                            .get(&format!("<0x{b:02X}>"))
                            .ok_or("missing byte fallback")?,
                    )
                }
            } else {
                return Err(format!("missing BPE piece {p}"));
            }
        }
        Ok(())
    }
    fn normal(&self, s: &str, out: &mut Vec<usize>) -> Result<(), String> {
        if self.qwen {
            return self.qwen_normal(s, out);
        }
        if self.gemma {
            let mut s = s.replace(' ', "▁");
            if !s.starts_with('▁') {
                s.insert(0, '▁')
            }
            let mut start = 0;
            while start < s.len() {
                let end = s[start + 3..]
                    .find('▁')
                    .map(|n| start + 3 + n)
                    .unwrap_or(s.len());
                self.bpe(s[start..end].chars().map(|c| c.to_string()).collect(), out)?;
                start = end
            }
            return Ok(());
        }
        // The upstream C++ tokenizer uses ASCII GPT-2 pretokenization, including
        // byte-oriented punctuation. Match it exactly, including whitespace lookahead.
        fn ws(c: u8) -> bool {
            matches!(c, b' ' | b'\t' | b'\r' | b'\n' | 11 | 12)
        }
        fn cat(c: u8) -> u8 {
            if c.is_ascii_alphabetic() {
                1
            } else if c.is_ascii_digit() {
                2
            } else if ws(c) {
                0
            } else {
                3
            }
        }
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let start = i;
            let mut end = None;
            if b[i] == b'\'' {
                for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
                    if b[i..].starts_with(c.as_bytes()) {
                        end = Some(i + c.len());
                        break;
                    }
                }
            }
            if end.is_none() {
                let j = if b[i] == b' ' && i + 1 < b.len() && !ws(b[i + 1]) {
                    i + 1
                } else {
                    i
                };
                let c = cat(b[j]);
                if c != 0 {
                    let mut k = j + 1;
                    while k < b.len() && cat(b[k]) == c {
                        k += 1
                    }
                    end = Some(k)
                } else {
                    let mut k = i + 1;
                    while k < b.len() && ws(b[k]) {
                        k += 1
                    }
                    if k < b.len() && k - i > 1 {
                        k -= 1
                    }
                    end = Some(k)
                }
            }
            i = end.unwrap();
            self.bpe(
                b[start..i]
                    .iter()
                    .map(|b| self.bytes[*b as usize].clone())
                    .collect(),
                out,
            )?;
        }
        Ok(())
    }

    fn qwen_normal(&self, text: &str, out: &mut Vec<usize>) -> Result<(), String> {
        use unicode_general_category::{get_general_category, GeneralCategory as C};
        use unicode_normalization::UnicodeNormalization;
        fn letter(c: char) -> bool {
            matches!(
                get_general_category(c),
                C::UppercaseLetter
                    | C::LowercaseLetter
                    | C::TitlecaseLetter
                    | C::ModifierLetter
                    | C::OtherLetter
            )
        }
        fn number(c: char) -> bool {
            matches!(
                get_general_category(c),
                C::DecimalNumber | C::LetterNumber | C::OtherNumber
            )
        }
        fn punct(c: char) -> bool {
            !c.is_whitespace() && !letter(c) && !number(c)
        }
        let normalized: String = text.nfc().collect();
        let cs: Vec<(usize, char)> = normalized.char_indices().collect();
        let mut i = 0;
        while i < cs.len() {
            let start = i;
            let mut end = None;
            if cs[i].1 == '\'' {
                for suffix in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
                    let bytes = &normalized.as_bytes()[cs[i].0..];
                    if bytes.len() >= suffix.len()
                        && bytes[..suffix.len()].eq_ignore_ascii_case(suffix.as_bytes())
                    {
                        end = Some(i + suffix.len());
                        break;
                    }
                }
            }
            if end.is_none() {
                let c = cs[i].1;
                let mut j = i;
                if !matches!(c, '\r' | '\n') && !letter(c) && !number(c) {
                    j += 1;
                }
                if j < cs.len() && letter(cs[j].1) {
                    j += 1;
                    while j < cs.len() && letter(cs[j].1) {
                        j += 1;
                    }
                    end = Some(j);
                }
            }
            if end.is_none() && number(cs[i].1) {
                end = Some(i + 1);
            }
            if end.is_none() {
                let mut j = i;
                if cs[j].1 == ' ' {
                    j += 1;
                }
                if j < cs.len() && punct(cs[j].1) {
                    j += 1;
                    while j < cs.len() && punct(cs[j].1) {
                        j += 1;
                    }
                    while j < cs.len() && matches!(cs[j].1, '\r' | '\n') {
                        j += 1;
                    }
                    end = Some(j);
                }
            }
            if end.is_none() && cs[i].1.is_whitespace() {
                let mut j = i;
                let mut newline = None;
                while j < cs.len() && cs[j].1.is_whitespace() {
                    if matches!(cs[j].1, '\r' | '\n') {
                        newline = Some(j + 1);
                    }
                    j += 1;
                }
                end =
                    Some(
                        newline
                            .unwrap_or_else(|| if j < cs.len() && j > i + 1 { j - 1 } else { j }),
                    );
            }
            i = end.ok_or("unsupported Qwen pretokenization character")?;
            let lo = cs[start].0;
            let hi = cs.get(i).map(|x| x.0).unwrap_or(normalized.len());
            self.bpe(
                normalized.as_bytes()[lo..hi]
                    .iter()
                    .map(|b| self.bytes[*b as usize].clone())
                    .collect(),
                out,
            )?;
        }
        Ok(())
    }

    pub fn encode(&self, s: &str) -> Result<Vec<usize>, String> {
        if s.len() > 1024 * 1024 {
            return Err("text exceeds 1 MiB".into());
        }
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < s.len() {
            let mut found: Option<(usize, usize, usize)> = None;
            for (t, id) in &self.special {
                if let Some(i) = s[pos..].find(t) {
                    let p = pos + i;
                    if found.map(|(old, _, _)| p < old).unwrap_or(true) {
                        found = Some((p, t.len(), *id))
                    }
                }
            }
            if let Some((i, len, id)) = found {
                self.normal(&s[pos..i], &mut out)?;
                out.push(id);
                pos = i + len
            } else {
                self.normal(&s[pos..], &mut out)?;
                break;
            }
        }
        Ok(out)
    }
}
