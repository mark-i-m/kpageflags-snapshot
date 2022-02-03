//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    collections::BTreeMap,
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
    Error = 1,
    Referenced = 2,
    Uptodate = 3,
    Dirty = 4,
    Lru = 5,
    Active = 6,
    Slab = 7,
    Writeback = 8,
    Reclaim = 9,
    Buddy = 10,
    Mmap = 11,
    Anon = 12,
    Swapcache = 13,
    Swapbacked = 14,
    CompoundHead = 15,
    CompoundTail = 16,
    Huge = 17,
    Unevictable = 18,
    Hwpoison = 19,
    Nopage = 20,
    Ksm = 21,
    Thp = 22,
    Offline = 23,
    ZeroPage = 24,
    Idle = 25,
    Pgtable = 26,

    MAX1,

    Reserved = 32,
    Mlocked = 33,
    Mappedtodisk = 34,
    Private = 35,
    Private2 = 36,
    OwnerPrivate = 37,
    Arch = 38,
    Uncached = 39,
    Softdirty = 40,
    Arch2 = 41,

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
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
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

impl From<KPF> for KPageFlags {
    fn from(kpf: KPF) -> Self {
        KPageFlags(kpf as u64)
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

    let mut stats = BTreeMap::new();

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
                    println!("{:010X}            {:8}KB {}", run_start, size, run_flags);
                } else {
                    let size = (pfn - run_start) * 4;
                    println!(
                        "{:010X}-{:010X} {:8}KB {}",
                        run_start,
                        pfn - 1,
                        size,
                        run_flags
                    );
                }

                *stats.entry(run_flags).or_insert(0) += pfn - run_start;

                run_start = pfn;
                run_flags = flags;
            }

            pfn += 1;
        }
    }

    // Print some stats about the different types of page usage.
    println!("SUMMARY");
    let mut total = 0;
    for (flags, npages) in stats.into_iter() {
        let size = npages * 4;
        if flags != KPageFlags::from(KPF::Nopage) {
            total += size;
        }

        let size = if size >= 1024 {
            format!("{:6}MB", size >> 10)
        } else {
            format!("{:6}KB", size)
        };
        println!("{} {}", size, flags);
    }

    println!("TOTAL: {}MB", total >> 10);

    Ok(())
}
