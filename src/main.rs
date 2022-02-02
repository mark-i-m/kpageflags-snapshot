//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    fs,
    io::{self, BufRead, ErrorKind},
};

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

// It's not actually dead code... the `KPF::from` function allows constructing all of them...
#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
enum KPF {
    Locked,
    Error,
    Referenced,
    Uptodate,
    Dirty,
    Lru,
    Active,
    Slab,
    Writeback,
    Reclaim,
    Buddy,
    Mmap,
    Anon,
    Swapcache,
    Swapbacked,
    CompoundHead,
    CompoundTail,
    Huge,
    Unevictable,
    Hwpoison,
    Nopage,
    Ksm,
    Thp,
    Balloon,
    ZeroPage,
    Idle,

    MAX,
}

impl KPF {
    pub fn from(val: u64) -> Self {
        assert!(val < (KPF::MAX as u64));
        unsafe { std::mem::transmute(val) }
    }
}

/// Represents the flags for a single physical page frame.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd)]
#[repr(transparent)]
struct KPageFlags(u64);

impl KPageFlags {
    /// Returns an empty set of flags.
    pub const fn empty() -> Self {
        KPageFlags(0)
    }

    /// Returns `true` if all bits in the given mask are set and `false` if any bits are not set.
    pub fn all(&self, mask: u64) -> bool {
        self.0 & mask == mask
    }
}

impl std::fmt::Display for KPageFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for fi in 0..(KPF::MAX as u64) {
            if self.all(1 << fi) {
                write!(f, "{:?} ", KPF::from(fi))?;
            }
        }

        writeln!(f)
    }
}

/// Wrapper around a `BufRead` type that for the `/proc/kpageflags` file.
struct KPageFlagsReader<B: BufRead> {
    buf_reader: B,
}

impl<B: BufRead> KPageFlagsReader<B> {
    pub fn new(buf_reader: B) -> Self {
        KPageFlagsReader { buf_reader }
    }

    /// Read exactly enough page flags to fill `buf` or return an error.
    pub fn read(&mut self, buf: &mut [KPageFlags]) -> io::Result<()> {
        // Cast as an array of bytes to do the read.
        let buf: &mut [u8] = unsafe {
            let ptr: *mut u8 = buf.as_mut_ptr() as *mut u8;
            let len = buf.len() * std::mem::size_of::<KPageFlags>();
            std::slice::from_raw_parts_mut(ptr, len)
        };

        self.buf_reader.read_exact(buf)?;

        Ok(())
    }
}

fn main() -> io::Result<()> {
    let file = fs::File::open(KPAGEFLAGS_PATH)?;
    let reader = io::BufReader::with_capacity(1 << 21 /* 2MB */, file);
    let mut flags = KPageFlagsReader::new(reader);

    let mut buf = vec![KPageFlags::empty(); 1 << (21 - 3)];
    let mut pfn = 0;

    let mut run_start = 0;
    let mut run_flags = KPageFlags::empty();

    loop {
        match flags.read(&mut buf) {
            Err(err) if matches!(err.kind(), ErrorKind::UnexpectedEof) => {
                break;
            }

            Err(err) => {
                panic!("{:?}", err);
            }

            Ok(()) => {}
        }

        for flags in buf.iter() {
            if pfn == 0 {
                run_flags = *flags;
            }

            if *flags != run_flags {
                if run_start == pfn - 1 {
                    print!("{:010X}            {}", run_start, run_flags);
                } else {
                    print!("{:010X}-{:010X} {}", run_start, pfn - 1, run_flags);
                }
                run_start = pfn;
                run_flags = *flags;
            }

            pfn += 1;
        }
    }

    Ok(())
}
