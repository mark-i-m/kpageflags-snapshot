//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    fs,
    io::{self, BufRead, ErrorKind},
};

const KPF_LOCKED: u64 = 1 << 0;
const KPF_ERROR: u64 = 1 << 1;
const KPF_REFERENCED: u64 = 1 << 2;
const KPF_UPTODATE: u64 = 1 << 3;
const KPF_DIRTY: u64 = 1 << 4;
const KPF_LRU: u64 = 1 << 5;
const KPF_ACTIVE: u64 = 1 << 6;
const KPF_SLAB: u64 = 1 << 7;
const KPF_WRITEBACK: u64 = 1 << 8;
const KPF_RECLAIM: u64 = 1 << 9;
const KPF_BUDDY: u64 = 1 << 10;
const KPF_MMAP: u64 = 1 << 11;
const KPF_ANON: u64 = 1 << 12;
const KPF_SWAPCACHE: u64 = 1 << 13;
const KPF_SWAPBACKED: u64 = 1 << 14;
const KPF_COMPOUND_HEAD: u64 = 1 << 15;
const KPF_COMPOUND_TAIL: u64 = 1 << 16;
const KPF_HUGE: u64 = 1 << 17;
const KPF_UNEVICTABLE: u64 = 1 << 18;
const KPF_HWPOISON: u64 = 1 << 19;
const KPF_NOPAGE: u64 = 1 << 20;
const KPF_KSM: u64 = 1 << 21;
const KPF_THP: u64 = 1 << 22;
const KPF_BALLOON: u64 = 1 << 23;
const KPF_ZERO_PAGE: u64 = 1 << 24;
const KPF_IDLE: u64 = 1 << 25;

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

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

    /// Returns `true` if any bits in the given mask are set and `false` if all bits are not set.
    pub fn any(&self, mask: u64) -> bool {
        self.0 & mask > 0
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
            println!("{} {:?}", pfn, flags);

            pfn += 1;
        }
    }

    Ok(())
}
