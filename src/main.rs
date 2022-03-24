//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    fs, io,
    ops::{BitOr, BitOrAssign},
    path::PathBuf,
    str::FromStr,
};

use clap::Parser;
use process::{map_and_summary, markov};
use read::KPageFlagsReader;

mod process;
mod read;

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

/// Easier to derive FromStr...
macro_rules! kpf {
    (pub enum KPF { $($name:ident $(= $val:literal)?),+ $(,)? }) => {
        // It's not actually dead code... the `KPF::from` function allows constructing all of them...
        #[allow(dead_code)]
        #[derive(Copy, Clone, Debug)]
        #[repr(u64)]
        pub enum KPF {
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

kpf! { pub enum KPF {
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

    pub fn valid_mask() -> u64 {
        let mut v = 0;
        for b in Self::values() {
            v |= 1 << b;
        }
        v
    }
}

/// Represents the flags for a single physical page frame.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct KPageFlags(u64);

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

    /// Returns `true` if the given KPF bit is set; `false` otherwise.
    pub fn has(&self, kpf: KPF) -> bool {
        self.all(1 << (kpf as u64))
    }

    /// Returns `true` if _consecutive_ regions with flags `first` and then `second` can be
    /// combined into one big region.
    pub fn can_combine(first: Self, second: Self) -> bool {
        // Combine identical sets of pages.
        if first == second {
            return true;
        }

        // Combine compound head and compound tail pages.
        if first.has(KPF::CompoundHead) && second.has(KPF::CompoundTail) {
            return true;
        }

        false
    }

    /// Clear all bits set in the `mask` from this `KPageFlags`.
    pub fn clear(&mut self, mask: u64) {
        self.0 &= !mask;
    }

    pub fn as_u64(self) -> u64 {
        self.0
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

        let invalid_bits = self.0 & !KPF::valid_mask();
        if invalid_bits != 0 {
            write!(f, "INVALID BITS: {invalid_bits:X?}")?;
        }

        Ok(())
    }
}

/// A program to process the contents of `/proc/kpageflags`.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// The path to the kpageflags. If not passed, use `/proc/kpageflags`.
    #[clap(short, long, default_value = KPAGEFLAGS_PATH)]
    file: PathBuf,

    /// Ignore the given flag. Effectively, treat all pages as if the flag is unset. This can be
    /// passed multiple times.
    #[clap(name = "FLAG", long = "ignore", multiple_occurrences(true))]
    ignored_flags: Vec<KPF>,

    /// Decompress before processing.
    #[clap(long)]
    gzip: bool,

    /// Print the Markov Process rather than the map and summary.
    #[clap(long)]
    markov: bool,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let file = fs::File::open(&args.file)?;
    let reader = io::BufReader::with_capacity(1 << 21 /* 2MB */, file);

    if args.gzip {
        let reader = flate2::bufread::MultiGzDecoder::new(reader);
        let reader = io::BufReader::with_capacity(1 << 21, reader);
        let reader = KPageFlagsReader::new(reader);
        if args.markov {
            markov(reader, &args)?;
        } else {
            map_and_summary(reader, &args)?;
        }
    } else {
        let reader = KPageFlagsReader::new(reader);
        if args.markov {
            markov(reader, &args)?;
        } else {
            map_and_summary(reader, &args)?;
        }
    }

    Ok(())
}
