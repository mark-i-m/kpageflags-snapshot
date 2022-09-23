//! Logic for processing kpageflags.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
    io::{self, Read, Write},
};

use encyclopagia::kpageflags::{Flaggy, KPageFlags, KPageFlagsIterator, KPageFlagsReader};
use hdrhistogram::Histogram;
use nalgebra::DMatrix;

use crate::Args;

/// The `MAX_ORDER` for Linux 5.17 (and a lot of older versions).
const MAX_ORDER: u64 = 10;

#[derive(Copy, Clone, Debug)]
pub struct CombinedPageFlags<K: Flaggy> {
    /// Starting PFN (inclusive).
    pub start: u64,
    /// Ending PFN (exlusive).
    pub end: u64,

    /// Flags of the indicated region.
    pub flags: KPageFlags<K>,
}

impl<K: Flaggy> Ord for CombinedPageFlags<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let combinedself = (self.end - self.start) ^ self.flags.as_u64();
        let combinedother = (other.end - other.start) ^ other.flags.as_u64();

        Ord::cmp(&combinedself, &combinedother)
    }
}
impl<K: Flaggy> PartialOrd for CombinedPageFlags<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl<K: Flaggy> PartialEq for CombinedPageFlags<K> {
    fn eq(&self, other: &Self) -> bool {
        matches!(Self::cmp(self, other), Ordering::Equal)
    }
}
impl<K: Flaggy> Eq for CombinedPageFlags<K> {}

impl<K: Flaggy> Hash for CombinedPageFlags<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash size and flags.
        (self.end - self.start).hash(state);
        self.flags.hash(state);
    }
}

/// Consumes an iterator over flags and transforms it to combine various elements and simplify
/// flags. This makes the stream a bit easier to plot and produce a markov process from.
pub struct KPageFlagsProcessor<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> {
    flags: std::iter::Peekable<std::iter::Enumerate<I>>,
}

impl<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> KPageFlagsProcessor<K, I> {
    pub fn new(iter: I) -> Self {
        Self {
            flags: iter.enumerate().peekable(),
        }
    }
}

impl<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> Iterator for KPageFlagsProcessor<K, I> {
    type Item = CombinedPageFlags<K>;

    fn next(&mut self) -> Option<Self::Item> {
        // Start with whatever the next flags are.
        let mut combined = {
            let (start, flags) = if let Some((start, flags)) = self.flags.next() {
                (start as u64, flags)
            } else {
                return None;
            };

            CombinedPageFlags {
                start,
                end: start + 1,
                flags,
            }
        };

        // Look ahead 1 element to see if we break the run...
        while let Some((_, next_flags)) = self.flags.peek() {
            // If this element can be combined with `combined`, combine it.
            // We want to limit the max size of a chunk to `MAX_ORDER`.
            if (combined.end - combined.start) < (1 << MAX_ORDER)
                && KPageFlags::can_combine(combined.flags, *next_flags)
            {
                let (pfn, flags) = self.flags.next().unwrap();
                combined.end = pfn as u64 + 1; // exclusive
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

/// Consumes a collection of iterators over flags in lockstep. In each call to `next`, it calls
/// `next` on each of the sub-iterators and returns an array with the returned values.
pub struct KPageFlagsLockstep<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> {
    iters: Vec<I>,
}

impl<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> KPageFlagsLockstep<K, I> {
    pub fn new(iters: Vec<I>) -> Self {
        KPageFlagsLockstep { iters }
    }
}

impl<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> Iterator for KPageFlagsLockstep<K, I> {
    type Item = Vec<Option<I::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        let out: Vec<Option<I::Item>> = self.iters.iter_mut().map(|i| i.next()).collect();

        if out.iter().all(Option::is_none) {
            None
        } else {
            Some(out)
        }
    }
}

pub fn map_and_summary<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    args: &Args,
) -> io::Result<()> {
    let flags = KPageFlagsProcessor::new(KPageFlagsIterator::new(reader, ignored_flags));
    let mut stats = BTreeMap::new();

    // Iterate over contiguous physical memory regions with similar properties.
    for region in flags {
        if args.flags {
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
        }

        let (total, stats) = stats
            .entry(region.flags)
            .or_insert_with(|| (0, Histogram::<u64>::new(5).unwrap()));

        *total += region.end - region.start;
        stats.record(region.end - region.start).unwrap();
    }

    // Print some stats about the different types of page usage.
    if args.summary {
        println!("SUMMARY");
        let mut total = 0;
        for (flags, (npages, stats)) in stats.into_iter() {
            let size = npages * 4;
            if flags != KPageFlags::from(K::NOPAGE) {
                total += size;
            }

            let size = if size >= 1024 {
                format!("{:6}MB", size >> 10)
            } else {
                format!("{size:6}KB")
            };

            let min = stats.min();
            let p25 = stats.value_at_quantile(0.25);
            let p50 = stats.value_at_quantile(0.50);
            let p75 = stats.value_at_quantile(0.75);
            let max = stats.max();

            println!("{min} {p25} {p50} {p75} {max} {size} {flags}");
        }

        println!("TOTAL: {}MB", total >> 10);
    }

    Ok(())
}

/// An iterator that returns (current, next) for all items in the stream.
struct PairIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    iter: I,
    prev: Option<<I as Iterator>::Item>,
}

impl<I> PairIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    pub fn new(iter: I) -> Self {
        Self { iter, prev: None }
    }
}

impl<I> Iterator for PairIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    type Item = (<I as Iterator>::Item, <I as Iterator>::Item);

    fn next(&mut self) -> Option<Self::Item> {
        if self.prev.is_none() {
            self.prev = self.iter.next();
        }

        if let Some(ref mut prev) = self.prev {
            if let Some(curr) = self.iter.next() {
                let old_prev = prev.clone();
                *prev = curr;
                Some((old_prev.clone(), prev.clone()))
            } else {
                None
            }
        } else {
            None
        }
    }
}

const FLAGS_BUDDY: u64 = 1 << 0;
const FLAGS_FILE: u64 = 1 << 1;
const FLAGS_ANON: u64 = 1 << 2;
const FLAGS_ANON_THP: u64 = 1 << 3;
const FLAGS_NONE: u64 = 1 << 4;
const FLAGS_PINNED: u64 = 1 << 5;

#[derive(Copy, Clone, Debug)]
struct CombinedGFPRegion {
    /// Order of allocation.
    pub order: u64,

    /// GFP for the given region.
    pub flags: u64,
}

impl Ord for CombinedGFPRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let combinedself = self.order ^ self.flags;
        let combinedother = other.order ^ other.flags;

        Ord::cmp(&combinedself, &combinedother)
    }
}
impl PartialOrd for CombinedGFPRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl PartialEq for CombinedGFPRegion {
    fn eq(&self, other: &Self) -> bool {
        matches!(Self::cmp(self, other), Ordering::Equal)
    }
}
impl Eq for CombinedGFPRegion {}

impl Hash for CombinedGFPRegion {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash size and flags.
        self.order.hash(state);
        self.flags.hash(state);
    }
}

/// Get the log (base 2) of `x`. `x` must be a power of two.
fn log2(x: u64) -> u64 {
    // Optimize the most common values we will get, then just do the computation.
    match x {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        128 => 7,
        256 => 8,
        512 => 9,
        1024 => 10,
        2048 => 11,
        4096 => 12,
        9182 => 13,
        other if other.is_power_of_two() => {
            for i in 0.. {
                if other >> i == 1 {
                    return i;
                }
            }

            unreachable!()
        }

        _ => panic!(),
    }
}

pub fn markov<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
) -> io::Result<()> {
    let flags = PairIterator::new(
        KPageFlagsProcessor::new(KPageFlagsIterator::new(reader, ignored_flags))
            .filter(|combined| !combined.flags.has(K::NOPAGE))
            .filter(|combined| !combined.flags.has(K::RESERVED))
            .map(|combined| CombinedGFPRegion {
                order: log2((combined.end - combined.start).next_power_of_two()),
                flags:
                    // Kernel memory.
                    if combined.flags.has(K::SLAB) {
                    FLAGS_PINNED
                } else if K::PGTABLE.is_some() && combined.flags.has(K::PGTABLE.unwrap()) {
                    FLAGS_PINNED
                }
                // Free pages.
                else if combined.flags.has(K::BUDDY) {
                    FLAGS_BUDDY
                }
                // Anonymous memory.
                else if combined.flags.has(K::ANON) && combined.flags.has(K::THP) {
                    FLAGS_ANON_THP
                }
                else if combined.flags.has(K::ANON) {
                    FLAGS_ANON
                }
                // File cache.
                else if combined.flags.has(K::LRU) {
                    FLAGS_FILE
                }
                // No flags... VM balloon drivers, IO buffers, etc.
                else {
                    FLAGS_NONE
                },
            }),
    );

    // graph[a][b] = number of edges from a -> b in the graph.
    let mut outnodes = HashSet::new();
    let mut innodes = HashSet::new();
    let mut graph = BTreeMap::new();

    for (fa, fb) in flags {
        *graph
            .entry(fa)
            .or_insert_with(BTreeMap::new)
            .entry(fb)
            .or_insert(0) += 1;

        outnodes.insert(fa);
        innodes.insert(fb);
    }

    // To simplify our lives later, we check for any graph nodes that have no outgoing edges. For
    // these, we add a self-loop edge.
    for sink in &innodes - &outnodes {
        *graph
            .entry(sink)
            .or_insert_with(BTreeMap::new)
            .entry(sink)
            .or_insert(0) += 1;
    }

    // Compute edge probabilities and output graph. Also construct probability transition matrix
    // `p` so that we can compute the stationary distribution later.
    let mut p = DMatrix::repeat(graph.len(), graph.len(), 0.0);
    for (i, (fa, out)) in graph.iter().enumerate() {
        let total_out = out.iter().map(|(_fb, count)| count).sum::<u64>() as f64;

        let order = fa.order;
        let flags = fa.flags;
        print!("{order} {flags:X}");

        // We need to make sure that the values add to 100. We do this by adding 1 to the rounded
        // probabilities that are largest until we make up the rounding error.
        let diff = 100
            - out
                .iter()
                .map(|(_fb, count)| ((*count as f64 / total_out * 100.0) as usize).clamp(0, 100))
                .sum::<usize>();
        let remainders = {
            let mut remainders = out
                .iter()
                .map(|(_fb, count)| (*count as f64 / total_out * 100.0).fract())
                .enumerate()
                .collect::<Vec<_>>();
            remainders.sort_by_key(|(_, fract)| (fract * 1000.0) as u64);
            remainders.truncate(diff);
            remainders.into_iter().collect::<HashMap<_, _>>()
        };

        for (j, (fb, count)) in out.iter().enumerate() {
            let idx = graph
                .keys()
                .enumerate()
                .find_map(|(i, f)| (*f == *fb).then(|| i))
                .unwrap();
            let prob = ((*count as f64 / total_out * 100.0) as u64).clamp(0, 100)
                + remainders.get(&j).map(|_| 1).unwrap_or(0);
            print!(" {idx} {prob}");

            p[(i, j)] = prob as f64 / 100.;
        }

        print!(";");

        io::stdout().flush()?;
    }

    // Compute stationary distribution of markov process. We can raise `p` to a large power and
    // then take any row.
    print!("\nStationary Distribution:");
    for pi in p.pow(1000).row(0).iter() {
        print!(" {pi:0.2}");
    }
    io::stdout().flush()?;

    Ok(())
}

pub fn type_dists<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
) -> io::Result<()> {
    let flags = KPageFlagsProcessor::new(KPageFlagsIterator::new(reader, ignored_flags));
    let mut stats = BTreeMap::new();

    // Iterate over contiguous physical memory regions with similar properties.
    for region in flags {
        let orders = stats
            .entry(region.flags)
            .or_insert_with(|| [0; MAX_ORDER as usize + 1]);
        let order = log2((region.end - region.start).next_power_of_two()) as usize;
        orders[order] += 1;
    }

    // Print some stats about the different types of page usage.
    for (flags, orders) in stats.into_iter() {
        for o in 0..orders.len() {
            print!("{} ", orders[o] << o);
        }
        println!("{flags}");
    }

    Ok(())
}

pub fn compare_snapshots<K: Flaggy>(ignored_flags: &[K], args: &Args) -> io::Result<()> {
    let snapshot_iterators = KPageFlagsLockstep::new(
        args.compare
            .iter()
            .map(|fname| {
                let reader: KPageFlagsReader<_, K> = super::open(args.gzip, fname).unwrap();
                KPageFlagsIterator::new(reader, ignored_flags)
            })
            .collect(),
    );

    for set_of_flags in snapshot_iterators {
        let mut changes = 0;
        let mut prev = set_of_flags[0];
        for flags in set_of_flags.iter().skip(1) {
            if *flags != prev {
                changes += 1;
            }
            prev = *flags;
        }

        println!("{changes}");
    }

    Ok(())
}
