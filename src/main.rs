//! Reads /proc/kpageflags on Linux 5.17 to snapshot the usage of system memory.

use std::{
    fs,
    io::{self, BufRead, BufReader, Read},
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::Parser;
use encyclopagia::kpageflags::{
    Flaggy, KPageFlagsReader, KPAGEFLAGS_PATH, KPF3_10_0, KPF4_15_0, KPF5_0_8, KPF5_13_0,
    KPF5_15_0, KPF5_17_0, KPF5_4_0,
};
use flate2::bufread::MultiGzDecoder;
use process::{compare_snapshots, map_and_summary, markov, type_dists};

mod process;

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

    /// Print the distribution of contiguity for each set of flags. This is the number of pages in
    /// each order of that type (similar to `/proc/buddyinfo`).
    #[clap(long)]
    dist: bool,

    /// Print the Markov Process.
    #[clap(long)]
    markov: bool,

    /// Compare the following snapshots in the given order (pass this flag multiple times) to see
    /// the change of in memory usage over time on a page by page basis.
    #[clap(
        long,
        required = false,
        multiple_occurrences = false,
        multiple_values = true
    )]
    compare: Vec<PathBuf>,

    /// Used for validating `superultramegafragmentor`: treat the PG_private, PG_private_2, and
    /// PG_reserved flags as if they were other more normal flags.
    #[clap(long)]
    simulated_flags: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum Kernel {
    V3_10_0,
    V4_15_0,
    V5_0_8,
    V5_4_0,
    V5_13_0,
    V5_15_0,
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
            "5.15.0" => Ok(Kernel::V5_15_0),
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

fn open<K: Flaggy>(
    gzip: bool,
    file: impl AsRef<Path>,
) -> std::io::Result<KPageFlagsReader<impl Read, K>> {
    let file = fs::File::open(&file)?;

    let reader = BufReader::with_capacity(1 << 21 /* 2MB */, file);
    let reader = if gzip {
        Adapter::Zipped(MultiGzDecoder::new(reader))
    } else {
        Adapter::Normal(reader)
    };
    let reader = BufReader::with_capacity(1 << 21, reader);
    Ok(KPageFlagsReader::new(reader))
}

fn process<K>(args: &Args) -> std::io::Result<()>
where
    K: Flaggy,
    <K as FromStr>::Err: std::fmt::Debug,
{
    let ignored_flags: Vec<K> = args
        .ignored_flags
        .iter()
        .map(|s| K::from_str(s).unwrap())
        .collect();

    if args.flags || args.summary {
        let reader = open(args.gzip, &args.file)?;
        map_and_summary(reader, &ignored_flags, &args)?;
    }
    if args.markov {
        let reader = open(args.gzip, &args.file)?;
        markov(reader, &ignored_flags, args.simulated_flags)?;
    }
    if args.dist {
        let reader = open(args.gzip, &args.file)?;
        type_dists(reader, &ignored_flags, args.simulated_flags)?;
    }

    if args.compare.len() > 0 {
        compare_snapshots::<K>(&ignored_flags, &args)?;
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    match args.kernel {
        Kernel::V3_10_0 => process::<KPF3_10_0::Flags>(&args),
        Kernel::V4_15_0 => process::<KPF4_15_0::Flags>(&args),
        Kernel::V5_0_8 => process::<KPF5_0_8::Flags>(&args),
        Kernel::V5_4_0 => process::<KPF5_4_0::Flags>(&args),
        Kernel::V5_13_0 => process::<KPF5_13_0::Flags>(&args),
        Kernel::V5_15_0 => process::<KPF5_15_0::Flags>(&args),
        Kernel::V5_17_0 => process::<KPF5_17_0::Flags>(&args),
    }
}
