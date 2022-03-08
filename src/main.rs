//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead},
    ops::{BitOr, BitOrAssign},
    path::PathBuf,
    str::FromStr,
};

use clap::Parser;

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

/// Easier to derive FromStr...
macro_rules! kpf {
    (enum KPF { $($name:ident $(= $val:literal)?),+ $(,)? }) => {
        // It's not actually dead code... the `KPF::from` function allows constructing all of them...
        #[allow(dead_code)]
        #[derive(Copy, Clone, Debug)]
        #[repr(u64)]
        enum KPF {
            $($name $(= $val)?),+
        }

        impl FromStr for KPF {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $(
                        stringify!($name) => Ok(KPF::$name),
                    )+

                    other => Err(format!("unknown flag: {}", other)),
                }
            }
        }
    };
}

kpf! { enum KPF {
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
}}

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

    /// Returns `true` if _consecutive_ regions with flags `first` and then `second` can be
    /// combined into one big region.
    pub fn can_combine(first: Self, second: Self) -> bool {
        // Combine identical sets of pages.
        if first == second {
            return true;
        }

        // Combine compound head and compound tail pages.
        if first.all(1 << (KPF::CompoundHead as u64)) && second.all(1 << (KPF::CompoundTail as u64))
        {
            return true;
        }

        false
    }

    /// Clear all bits set in the `mask` from this `KPageFlags`.
    pub fn clear(&mut self, mask: u64) {
        self.0 &= !mask;
    }
}

impl BitOr for KPageFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        KPageFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for KPageFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl From<KPF> for KPageFlags {
    fn from(kpf: KPF) -> Self {
        KPageFlags(1 << (kpf as u64))
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

/// Turns a `KPageFlagsReader` into a proper (efficient) iterator over flags.
struct KPageFlagsIterator<B: BufRead> {
    /// The reader we are reading from.
    reader: KPageFlagsReader<B>,

    /// Temporary buffer for data read but not consumed yet.
    buf: [KPageFlags; 1 << (21 - 3)],
    /// The number of valid flags in the buffer.
    nflags: usize,
    /// The index of the first valid, unconsumed flag in the buffer, if `nflags > 0`.
    idx: usize,
}

impl<B: BufRead> KPageFlagsIterator<B> {
    pub fn new(reader: KPageFlagsReader<B>) -> Self {
        KPageFlagsIterator {
            reader,
            buf: [KPageFlags::empty(); 1 << (21 - 3)],
            nflags: 0,
            idx: 0,
        }
    }
}

impl<B: BufRead> Iterator for KPageFlagsIterator<B> {
    type Item = KPageFlags;

    fn next(&mut self) -> Option<Self::Item> {
        // Need to read some more?
        if self.nflags == 0 {
            self.nflags = match self.reader.read(&mut self.buf) {
                Err(err) => {
                    panic!("{:?}", err);
                }

                // EOF
                Ok(0) => return None,

                Ok(nflags) => nflags,
            };
            self.idx = 0;
        }

        // Return the first valid flags.
        let item = self.buf[self.idx];

        self.nflags -= 1;
        self.idx += 1;

        Some(item)
    }
}

struct CombinedPageFlags {
    /// Starting PFN (inclusive).
    pub start: u64,
    /// Ending PFN (exlusive).
    pub end: u64,

    /// Flags of the indicated region.
    pub flags: KPageFlags,
}

/// Consumes an iterator over flags and transforms it to combine various elements and simplify
/// flags. This makes the stream a bit easier to plot and produce a markov process from.
struct KPageFlagsProcessor<I: Iterator<Item = KPageFlags>> {
    flags: std::iter::Peekable<std::iter::Enumerate<I>>,
    ignored_flags: u64,
}

impl<I: Iterator<Item = KPageFlags>> KPageFlagsProcessor<I> {
    pub fn new(iter: I, ignored_flags: Vec<KPF>) -> Self {
        Self {
            flags: iter.enumerate().peekable(),
            ignored_flags: {
                let mut mask = 0;

                for f in ignored_flags.into_iter() {
                    mask |= 1 << (f as u64);
                }

                mask
            },
        }
    }
}

impl<I: Iterator<Item = KPageFlags>> Iterator for KPageFlagsProcessor<I> {
    type Item = CombinedPageFlags;

    fn next(&mut self) -> Option<Self::Item> {
        // Start with whatever the next flags are.
        let mut combined = {
            let (start, mut flags) = if let Some((start, flags)) = self.flags.next() {
                (start as u64, flags)
            } else {
                return None;
            };

            flags.clear(self.ignored_flags);

            CombinedPageFlags {
                start,
                end: start + 1,
                flags,
            }
        };

        // Look ahead 1 element to see if we break the run...
        while let Some((_, next_flags)) = self.flags.peek() {
            let mut next_flags = *next_flags;
            next_flags.clear(self.ignored_flags);

            // If this element can be combined with `combined`, combine it.
            if KPageFlags::can_combine(combined.flags, next_flags) {
                let (pfn, mut flags) = self.flags.next().unwrap();
                combined.end = pfn as u64 + 1; // exclusive
                flags.clear(self.ignored_flags);
                combined.flags |= flags;
            }
            // Otherwise, end the run and return here.
            else {
                break;
            }
        }

        Some(combined)
    }
}

/// A program to process the contents of `/proc/kpageflags`.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path to the kpageflags. If not passed, use `/proc/kpageflags`.
    #[clap(short, long, default_value = KPAGEFLAGS_PATH)]
    file: PathBuf,

    /// Ignore the given flag. Effectively, treat all pages as if the flag is unset. This can be
    /// passed multiple times.
    #[clap(name = "FLAG", long = "ignore", multiple_occurrences(true))]
    ignored_flags: Vec<KPF>,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let file = fs::File::open(args.file)?;
    let reader = io::BufReader::with_capacity(1 << 21 /* 2MB */, file);
    let flags = KPageFlagsProcessor::new(
        KPageFlagsIterator::new(KPageFlagsReader::new(reader)),
        args.ignored_flags,
    );

    let mut stats = BTreeMap::new();

    // Iterate over contiguous physical memory regions with similar properties.
    for region in flags {
        if region.end == region.start + 1 {
            // The only reason we specify `size` here is so that `println` can handle the width of
            // the field, so all the columns line up when we print...
            println!(
                "{:010X}            {:8}KB {}",
                region.start, 4, region.flags
            );
        } else {
            let size = (region.end - region.start) * 4;
            println!(
                "{:010X}-{:010X} {:8}KB {}",
                region.start, region.end, size, region.flags,
            );
        }

        *stats.entry(region.flags).or_insert(0) += region.end - region.start;
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
