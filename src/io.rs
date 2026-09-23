use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use goish::syscall;
pub fn write(fd: i32, s: &str) {
    let mut b = s.as_bytes();
    while !b.is_empty() {
        let n = syscall::Write(fd, b.as_ptr(), b.len());
        if n <= 0 {
            break;
        }
        b = &b[n as usize..];
    }
}
pub fn out(s: &str) {
    write(1, s);
    write(1, "\n");
}
pub fn log(s: &str) {
    write(2, s);
    write(2, "\n");
}
pub fn read(path: &str) -> Result<String, String> {
    let (v, e) = goish::os::ReadFile(path);
    if e != goish::nil {
        return Err(crate::format!("cannot read {path}: {e:?}"));
    }
    String::from_utf8(v.as_ref().to_vec()).map_err(|_| "file is not UTF-8".to_string())
}
pub fn args() -> Vec<String> {
    let a = goish::os::Args();
    (0..a.Len()).map(|i| a[i].to_string()).collect()
}
pub fn line() -> Option<String> {
    let mut v = Vec::new();
    loop {
        let mut b = 0;
        let n = syscall::Read(0, &mut b, 1);
        if n <= 0 {
            return if v.is_empty() {
                None
            } else {
                String::from_utf8(v).ok()
            };
        }
        if b == b'\n' {
            return String::from_utf8(v).ok();
        }
        v.push(b);
        if v.len() > 1048576 {
            return None;
        }
    }
}
pub struct Mapping {
    p: *mut u8,
    n: usize,
}
// SAFETY: the mapping is PROT_READ/MAP_PRIVATE, exposes only shared slices,
// and is unmapped only when its owner drops after all Arc<App> users finish.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
impl Mapping {
    pub fn open(path: &str) -> Result<Self, String> {
        let mut p = path.as_bytes().to_vec();
        if p.contains(&0) {
            return Err("NUL in path".into());
        }
        p.push(0);
        let fd = syscall::Open(p.as_ptr(), 0, 0);
        if fd < 0 {
            return Err(crate::format!("open {path}: errno {}", -fd));
        }
        let mut st = syscall::Stat_t::default();
        let rc = syscall::Fstat(fd, &mut st);
        if rc < 0 || st.st_size <= 0 {
            syscall::Close(fd);
            return Err("empty/unreadable model".into());
        }
        let n = st.st_size as usize;
        let p = syscall::Mmap(core::ptr::null_mut(), n, 1, 2, fd, 0);
        syscall::Close(fd);
        if (p as isize) < 0 && (p as isize) >= -4095 {
            return Err("mmap failed".into());
        }
        Ok(Self { p, n })
    }
    pub fn bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.p, self.n) }
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        syscall::Munmap(self.p, self.n);
    }
}
