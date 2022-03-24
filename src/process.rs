//! Logic for processing kpageflags.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
    io::{self, Read, Write},
};

use crate::{
    read::{KPageFlagsIterator, KPageFlagsReader},
    Args, KPageFlags, KPF,
};

/// The `MAX_ORDER` for Linux 5.17 (and a lot of older versions).
const MAX_ORDER: u64 = 10;

#[derive(Copy, Clone, Debug)]
pub struct CombinedPageFlags {
    /// Starting PFN (inclusive).
    pub start: u64,
    /// Ending PFN (exlusive).
    pub end: u64,

    /// Flags of the indicated region.
    pub flags: KPageFlags,
}

impl Ord for CombinedPageFlags {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let combinedself = (self.end - self.start) ^ self.flags.as_u64();
        let combinedother = (other.end - other.start) ^ other.flags.as_u64();

        Ord::cmp(&combinedself, &combinedother)
    }
}
impl PartialOrd for CombinedPageFlags {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Self::cmp(self, other))
    }
}

impl PartialEq for CombinedPageFlags {
    fn eq(&self, other: &Self) -> bool {
        matches!(Self::cmp(self, other), Ordering::Equal)
    }
}
impl Eq for CombinedPageFlags {}

impl Hash for CombinedPageFlags {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash size and flags.
        (self.end - self.start).hash(state);
        self.flags.hash(state);
    }
}

/// Consumes an iterator over flags and transforms it to combine various elements and simplify
/// flags. This makes the stream a bit easier to plot and produce a markov process from.
pub struct KPageFlagsProcessor<I: Iterator<Item = KPageFlags>> {
    flags: std::iter::Peekable<std::iter::Enumerate<I>>,
}

impl<I: Iterator<Item = KPageFlags>> KPageFlagsProcessor<I> {
    pub fn new(iter: I) -> Self {
        Self {
            flags: iter.enumerate().peekable(),
        }
    }
}

impl<I: Iterator<Item = KPageFlags>> Iterator for KPageFlagsProcessor<I> {
    type Item = CombinedPageFlags;

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

pub fn map_and_summary<R: Read>(reader: KPageFlagsReader<R>, args: &Args) -> io::Result<()> {
    let flags = KPageFlagsProcessor::new(KPageFlagsIterator::new(reader, &args.ignored_flags));
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
            format!("{size:6}KB")
        };
        println!("{size} {flags}");
    }

    println!("TOTAL: {}MB", total >> 10);

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

const GFP_KERNEL: u64 = 1 << 0;
const GFP_USER: u64 = 1 << 1;
const GFP_BUDDY: u64 = 1 << 2;

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

pub fn markov<R: Read>(reader: KPageFlagsReader<R>, args: &Args) -> io::Result<()> {
    let flags = PairIterator::new(
        KPageFlagsProcessor::new(KPageFlagsIterator::new(reader, &args.ignored_flags))
            .filter(|combined| !combined.flags.has(KPF::Nopage))
            .filter(|combined| !combined.flags.has(KPF::Reserved))
            .map(|combined| CombinedGFPRegion {
                order: log2((combined.end - combined.start).next_power_of_two()),
                flags: if combined.flags.has(KPF::Slab) || combined.flags.has(KPF::Pgtable) {
                    GFP_KERNEL
                } else if combined.flags.has(KPF::Mmap) {
                    GFP_USER
                }
                // Free pages...
                else if combined.flags.has(KPF::Buddy) {
                    GFP_BUDDY
                }
                // And none of the above, but clearly not being used by user.
                else if combined.flags.has(KPF::Lru) {
                    GFP_KERNEL
                }
                // No flags...
                else {
                    GFP_KERNEL
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

    println!("MARKOV");

    // Compute edge probabilities and output graph.
    for (fa, out) in graph.iter() {
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

        for (i, (fb, count)) in out.iter().enumerate() {
            let idx = graph
                .keys()
                .enumerate()
                .find_map(|(i, f)| (*f == *fb).then(|| i))
                .unwrap();
            let prob = ((*count as f64 / total_out * 100.0) as u64).clamp(0, 100)
                + remainders.get(&i).map(|_| 1).unwrap_or(0);
            print!(" {idx} {prob}");
        }

        print!(";");

        io::stdout().flush()?;
    }

    Ok(())
}
