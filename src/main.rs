//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead},
};

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

// It's not actually dead code... the `KPF::from` function allows constructing all of them...
#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
#[repr(u64)]
enum KPF {
    Locked = 0,
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
    Offline,
    ZeroPage,
    Idle,
    Pgtable,

    MAX1,

    Reserved = 32,
    Mlocked,
    Mappedtodisk,
    Private,
    Private2,
    OwnerPrivate,
    Arch,
    Uncached,
    Softdirty,
    Arch2,

    MAX2,
}

impl KPF {
    pub fn from(val: u64) -> Self {
        assert!(Self::valid(val));
        unsafe { std::mem::transmute(val) }
    }

    pub fn valid(val: u64) -> bool {
        ((KPF::Locked as u64) <= val && val < (KPF::MAX1 as u64))
            || ((KPF::Reserved as u64) <= val && val < (KPF::MAX2 as u64))
    }

    pub fn values() -> impl Iterator<Item = u64> {
        ((KPF::Locked as u64)..(KPF::MAX1 as u64)).chain((KPF::Reserved as u64)..(KPF::MAX2 as u64))
    }
}

/// Represents the flags for a single physical page frame.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd)]
#[repr(transparent)]
struct KPageFlags(u64);

const KPF_SIZE: usize = std::mem::size_of::<KPageFlags>();

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
        for fi in KPF::values() {
            if self.all(1 << fi) {
                write!(f, "{:?} ", KPF::from(fi))?;
            }
        }

        Ok(())
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

    /// Similar to `Read::read`, but reads the bytes as `KPageFlags`, and returns the number of
    /// flags in the buffer, rather than the number of bytes.
    pub fn read(&mut self, buf: &mut [KPageFlags]) -> io::Result<usize> {
        // Cast as an array of bytes to do the read.
        let buf: &mut [u8] = unsafe {
            let ptr: *mut u8 = buf.as_mut_ptr() as *mut u8;
            let len = buf.len() * KPF_SIZE;
            std::slice::from_raw_parts_mut(ptr, len)
        };

        self.buf_reader.read(buf).map(|bytes| {
            assert_eq!(bytes % KPF_SIZE, 0);
            bytes / KPF_SIZE
        })
    }
}

fn main() -> io::Result<()> {
    let file = fs::File::open(KPAGEFLAGS_PATH)?;
    let reader = io::BufReader::with_capacity(1 << 21 /* 2MB */, file);
    let mut flags = KPageFlagsReader::new(reader);

    let mut stats = HashMap::new();

    let mut buf = vec![KPageFlags::empty(); 1 << (21 - 3)];
    let mut pfn = 0;
    let mut run_start = 0;
    let mut run_flags = KPageFlags::empty();

    loop {
        let nflags = match flags.read(&mut buf) {
            Err(err) => {
                panic!("{:?}", err);
            }

            // EOF
            Ok(0) => break,

            Ok(nflags) => nflags,
        };

        for flags in buf.iter().take(nflags) {
            let flags = *flags;

            if pfn == 0 {
                run_flags = flags;
            }

            if flags != run_flags {
                if run_start == pfn - 1 {
                    let size = 4; // KB
                    println!("{:010X}            {:5}KB {}", run_start, size, run_flags);
                } else {
                    let size = (pfn - 1 - run_start) * 4;
                    println!(
                        "{:010X}-{:010X} {:5}KB {}",
                        run_start,
                        pfn - 1,
                        size,
                        run_flags
                    );
                }

                *stats.entry(flags).or_insert(0) += pfn - 1 - run_start;

                run_start = pfn;
                run_flags = flags;
            }

            pfn += 1;
        }
    }

    // Print some stats about the different types of page usage.
    for (flags, npages) in stats.into_iter() {
        println!("{:7}KB {}", npages, flags);
    }

    Ok(())
}
