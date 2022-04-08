//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    fs,
    io::{self, BufRead, BufReader, Read},
    marker::PhantomData,
    ops::{BitOr, BitOrAssign},
    path::PathBuf,
    str::FromStr,
};

use clap::Parser;
use flate2::bufread::MultiGzDecoder;
use process::{map_and_summary, markov};
use read::KPageFlagsReader;

use crate::flags::{Flaggy, KPF5_17_0};

mod flags;
mod process;
mod read;

const KPAGEFLAGS_PATH: &str = "/proc/kpageflags";

/// Represents the flags for a single physical page frame.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct KPageFlags<K: Flaggy>(u64, PhantomData<K>);

const KPF_SIZE: usize = std::mem::size_of::<u64>();

impl<K: Flaggy> KPageFlags<K> {
    /// Returns an empty set of flags.
    pub fn empty() -> Self {
        KPageFlags(0, PhantomData)
    }

    /// Returns `true` if all bits in the given mask are set and `false` if any bits are not set.
    pub fn all(&self, mask: u64) -> bool {
        self.0 & mask == mask
    }

    /// Returns `true` if the given KPF bit is set; `false` otherwise.
    pub fn has(&self, kpf: K) -> bool {
        self.all(1 << kpf.into())
    }

    /// Returns `true` if _consecutive_ regions with flags `first` and then `second` can be
    /// combined into one big region.
    pub fn can_combine(first: Self, second: Self) -> bool {
        // Combine identical sets of pages.
        if first == second {
            return true;
        }

        // Combine compound head and compound tail pages.
        if first.has(K::COMPOUND_HEAD) && second.has(K::COMPOUND_TAIL) {
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

impl<K: Flaggy> BitOr for KPageFlags<K> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        KPageFlags(self.0 | rhs.0, PhantomData)
    }
}

impl<K: Flaggy> BitOrAssign for KPageFlags<K> {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl<K: Flaggy> From<K> for KPageFlags<K> {
    fn from(kpf: K) -> Self {
        KPageFlags(1 << kpf.into(), PhantomData)
    }
}

impl<K: Flaggy> std::fmt::Display for KPageFlags<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for fi in K::values() {
            if self.all(1 << fi) {
                write!(f, "{:?} ", K::from(*fi))?;
            }
        }

        let invalid_bits = self.0 & !K::valid_mask();
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
    ignored_flags: Vec<String>,

    /// Kernel version,
    #[clap(long)]
    kernel: Kernel,

    /// Decompress before processing.
    #[clap(long)]
    gzip: bool,

    /// List all page flags.
    #[clap(long)]
    flags: bool,

    /// Print a summary of page usages.
    #[clap(long)]
    summary: bool,

    /// Print the Markov Process.
    #[clap(long)]
    markov: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum Kernel {
    V3_10_0,
    V4_15_0,
    V5_0_8,
    V5_4_0,
    V5_13_0,
    V5_17_0,
}

impl FromStr for Kernel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "3.10.0" => Ok(Kernel::V3_10_0),
            "4.15.0" => Ok(Kernel::V4_15_0),
            "5.0.8" => Ok(Kernel::V5_0_8),
            "5.4.0" => Ok(Kernel::V5_4_0),
            "5.13.0" => Ok(Kernel::V5_13_0),
            "5.17.0" => Ok(Kernel::V5_17_0),

            other => Err(format!("Unknown kernel version: {other}")),
        }
    }
}

enum Adapter<R, B> {
    Zipped(MultiGzDecoder<B>),
    Normal(R),
}

impl<R: Read, B: BufRead> Read for Adapter<R, B> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Adapter::Zipped(r) => r.read(buf),
            Adapter::Normal(r) => r.read(buf),
        }
    }
}

fn parse_ignored_flags(args: &Args) -> Vec<impl Flaggy> {
    fn inner<K>(args: &Args) -> Vec<K>
    where
        K: Flaggy,
        <K as FromStr>::Err: std::fmt::Debug,
    {
        args.ignored_flags
            .iter()
            .map(|s| K::from_str(s).unwrap())
            .collect()
    }

    match args.kernel {
        Kernel::V3_10_0 => todo!(),
        Kernel::V4_15_0 => todo!(),
        Kernel::V5_0_8 => todo!(),
        Kernel::V5_4_0 => todo!(),
        Kernel::V5_13_0 => todo!(),
        Kernel::V5_17_0 => inner::<KPF5_17_0>(args),
    }
}

fn open<K: Flaggy>(args: &Args) -> std::io::Result<KPageFlagsReader<impl Read, K>> {
    let file = fs::File::open(&args.file)?;

    let reader = BufReader::with_capacity(1 << 21 /* 2MB */, file);
    let reader = if args.gzip {
        Adapter::Zipped(MultiGzDecoder::new(reader))
    } else {
        Adapter::Normal(reader)
    };
    let reader = BufReader::with_capacity(1 << 21, reader);
    Ok(KPageFlagsReader::new(reader))
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let ignored_flags = parse_ignored_flags(&args);

    if args.flags || args.summary {
        let reader = open(&args)?;
        map_and_summary(reader, &ignored_flags, &args)?;
    }
    if args.markov {
        let reader = open(&args)?;
        markov(reader, &ignored_flags)?;
    }

    Ok(())
}
