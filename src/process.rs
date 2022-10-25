//! Logic for processing kpageflags.

use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt::Display,
    hash::Hash,
    io::{self, Read},
};

use encyclopagia::{
    kpageflags::{Flaggy, KPageFlags, KPageFlagsIterator, KPageFlagsReader},
    FileReadableReader,
};
use hdrhistogram::Histogram;
use nalgebra::{DMatrix, DVector};

use crate::Args;

/// The `MAX_ORDER` for Linux 5.17 (and a lot of older versions).
pub const MAX_ORDER: u64 = 10;

/// The  granularity with which probabilities are expressed in MPs.
pub const MP_GRANULARITY: f64 = 1000.0;

#[derive(Copy, Clone, Debug)]
pub struct CombinedPageFlags<K: Flaggy> {
    /// Starting PFN (inclusive).
    pub start: u64,
    /// Ending PFN (exlusive).
    pub end: u64,

    /// Flags of the indicated region.
    pub flags: KPageFlags<K>,
}

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
    simulated_flags: bool,
}

impl<K: Flaggy, I: Iterator<Item = KPageFlags<K>>> KPageFlagsProcessor<K, I> {
    pub fn new(iter: I, simulated_flags: bool) -> Self {
        Self {
            flags: iter.enumerate().peekable(),
            simulated_flags,
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

        if self.simulated_flags && combined.flags.all(K::OWNERPRIVATE1 | K::RESERVED) {
            combined.flags.clear(K::OWNERPRIVATE1 | K::RESERVED);

            // Anon THP pages.
            if combined.flags.all(K::PRIVATE | K::PRIVATE2) {
                combined.flags.clear(K::PRIVATE | K::PRIVATE2);
                combined.flags |= KPageFlags::from(K::ANON | K::THP);
            }
            // Anon pages.
            else if combined.flags.all(K::PRIVATE) {
                combined.flags.clear(K::PRIVATE);
                combined.flags |= KPageFlags::from(K::ANON);
            }
            // File pages.
            else if combined.flags.all(K::LRU) {
                // Nothing to do...
            }
            // Private 2 without Private Cannot happen!
            else if combined.flags.all(K::PRIVATE2) {
                unreachable!();
            }
            // Pinned pages.
            else {
                combined.flags |= KPageFlags::from(K::SLAB);
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
    let flags = KPageFlagsProcessor::new(
        KPageFlagsIterator::new(reader, ignored_flags),
        args.simulated_flags,
    );
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
    pub fn new(mut iter: I) -> Self {
        let prev = iter.next();
        Self { iter, prev }
    }
}

impl<I> Iterator for PairIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    type Item = (<I as Iterator>::Item, <I as Iterator>::Item);

    fn next(&mut self) -> Option<Self::Item> {
        if let (Some(prev), Some(curr)) = (self.prev.take(), self.iter.next()) {
            self.prev = Some(curr.clone());
            Some((prev, curr))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u64)]
pub enum GFPFlags {
    Buddy = 1 << 0,
    File = 1 << 1,
    Anon = 1 << 2,
    AnonThp = 1 << 3,
    None = 1 << 4,
    Pinned = 1 << 5,
}

impl Display for GFPFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match self {
            GFPFlags::Buddy => 'B',
            GFPFlags::File => 'F',
            GFPFlags::Anon => 'A',
            GFPFlags::AnonThp => 'T',
            GFPFlags::None => 'N',
            GFPFlags::Pinned => 'P',
        };
        write!(f, "{ty}")
    }
}

#[derive(Copy, Clone, Debug)]
struct CombinedGFPRegion {
    /// Order of allocation.
    pub order: u64,

    /// GFP for the given region.
    pub flags: GFPFlags,
}

impl Ord for CombinedGFPRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let flagscmp = Ord::cmp(&self.flags, &other.flags);
        let ordercmp = Ord::cmp(&self.order, &other.order);

        if flagscmp == Ordering::Equal {
            ordercmp
        } else {
            flagscmp
        }
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

impl Display for CombinedGFPRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.flags, self.order)
    }
}

/// Get the log (base 2) of `x`. `x` must be a power of two.
#[allow(dead_code)]
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

pub fn kpf_to_abstract_flags<K: Flaggy>(flags: KPageFlags<K>) -> GFPFlags {
    // Kernel memory.
    if flags.all(K::SLAB) {
        GFPFlags::Pinned
    } else if K::PGTABLE.is_some() && flags.all(K::PGTABLE.unwrap()) {
        GFPFlags::Pinned
    }
    // Free pages.
    else if flags.all(K::BUDDY) {
        GFPFlags::Buddy
    }
    // Anonymous memory.
    else if flags.all(K::ANON | K::THP) {
        GFPFlags::AnonThp
    } else if flags.all(K::ANON) {
        GFPFlags::Anon
    }
    // File cache.
    else if flags.all(K::LRU) {
        GFPFlags::File
    }
    // No flags... VM balloon drivers, IO buffers, etc.
    else {
        GFPFlags::None
    }
}

/// Given a size `size`, break into the longest set of power-of-2-sized chunks less than
/// `MAX_ORDER` as possible.
fn break_into_pow_of_2_regions(size: u64) -> Vec<u64> {
    // The set of orders we want is already represented in the binary representation of `size`.

    const MAX_ORDER_MAX_LO: u64 = (1 << MAX_ORDER) - 1;
    const MAX_ORDER_MAX_HI: u64 = !MAX_ORDER_MAX_LO;

    let n_max_order = ((size & MAX_ORDER_MAX_HI) >> MAX_ORDER) as usize;

    let mut regions = vec![MAX_ORDER; n_max_order];

    for o in 0..(MAX_ORDER) {
        if size & (1 << o) != 0 {
            regions.push(o);
        }
    }

    regions
}

fn combine_and_clean_flags<K: Flaggy>(
    reader: FileReadableReader<impl Read, KPageFlags<K>>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> impl Iterator<Item = CombinedGFPRegion> {
    KPageFlagsProcessor::new(
        KPageFlagsIterator::new(reader, ignored_flags),
        simulated_flags,
    )
    .filter(|combined| !combined.flags.any(K::NOPAGE))
    .filter(|combined| !combined.flags.any(K::RESERVED))
    .flat_map(|combined| {
        break_into_pow_of_2_regions(combined.end - combined.start)
            .into_iter()
            .map(move |order| CombinedGFPRegion {
                order,
                flags: kpf_to_abstract_flags(combined.flags),
            })
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Reachability {
    Reachable,
    Unreachable,
}

/// Represents a Markov Process with the ability to cull unlikely states and check irreducibility.
struct MarkovProcess {
    // The main representation of the process is via the transition probability matrix. The matrix
    // shows the weight of each transition. We also keep a list of labels corresponding to the
    // different indices of the matrix. Both the label list and the matrix are in the same order.
    /// Transition probability matrix `p`: `p[i,j]` is the probability of going from `i` to `j` in
    /// one step. `labels[i]` is the label of `i`.
    p: DMatrix<f64>,

    /// Node labels.
    labels: Vec<CombinedGFPRegion>,
}

impl MarkovProcess {
    /// Construct a MP from the given iterator.
    pub fn construct(flags: impl Iterator<Item = (CombinedGFPRegion, CombinedGFPRegion)>) -> Self {
        // graph[a][b] = number of edges from a -> b in the graph.
        let mut graph = BTreeMap::new();
        for (fa, fb) in flags {
            // Add the edge...
            *graph
                .entry(fa)
                .or_insert_with(BTreeMap::new)
                .entry(fb)
                .or_insert(0.0) += 1.0;

            // And make sure both nodes are in the graph.
            graph.entry(fb).or_insert_with(BTreeMap::new);
        }

        // Compute a canonical ordered list of all nodes.
        let mut labels = graph.keys().cloned().collect::<Vec<_>>();
        labels.sort();

        // Construct probability transition matrix.
        let mut p = DMatrix::repeat(labels.len(), labels.len(), 0.0);
        for (i, fa) in labels.iter().enumerate() {
            let outgoing = &graph[&fa];
            let total_out = outgoing.values().sum::<f64>();

            for (j, fb) in labels.iter().enumerate() {
                let count = outgoing.get(&fb).unwrap_or(&0.0);
                let prob = count / total_out;
                p[(i, j)] = prob;
            }
        }

        MarkovProcess { p, labels }
    }

    /// Renormalize the probabilities in a row to ensure that they add to 1.
    fn renormalize_row(&mut self, node: usize) {
        let total = self.p.row(node).sum();
        for to in 0..self.p.ncols() {
            self.p[(node, to)] /= total;
        }
    }

    /// Remove the given nodes from the MP.
    fn remove_nodes(&mut self, to_remove: &[usize]) {
        // Sort in reverse order to avoid removals messing up the indices.
        let mut to_remove = to_remove.to_owned();
        to_remove.sort_by_key(|k| -(*k as i64));

        let new_p = self
            .p
            .clone()
            .remove_rows_at(&to_remove)
            .remove_columns_at(&to_remove);
        self.p = new_p;

        for node in to_remove.into_iter() {
            self.labels.remove(node);
        }
    }

    /// Cull unlikely edges and nodes from the MP. An edge `a->b` is removed if it has a
    /// probability less than `tol`. In this case, the other edges of `a` are renormalized so that
    /// their probabilities sum to 1 before any more edges are culled. `b` is removed if it has no
    /// incoming edges after culling `a->b`.
    pub fn cull_unlikely(&mut self, tol: f64) {
        loop {
            // Remove at most one edge from each node that has probability < `tol`.
            let mut culled_nodes = Vec::new();
            for from in 0..self.p.nrows() {
                if let Some(to) = (0..self.p.ncols())
                    .find(|&to| f64::EPSILON < self.p[(from, to)] && self.p[(from, to)] < tol)
                {
                    self.p[(from, to)] = 0.0;
                    culled_nodes.push(from);
                }
            }

            // If no edges were removed, we reached a fixed point.
            if culled_nodes.is_empty() {
                return;
            };

            // Renormalize remaining edges on each nodes so they sum to 1.
            for &node in culled_nodes.iter() {
                self.renormalize_row(node);
            }

            // If a node has no incoming edges, remove it. We go in reverse order so as not to mess
            // up indexing as we remove elements.
            let mut to_remove: Vec<usize> = Vec::new();
            for node in 0..self.p.ncols() {
                let total_incoming = self.p.column(node).sum();
                if total_incoming < f64::EPSILON {
                    to_remove.push(node);
                }
            }
            self.remove_nodes(&to_remove);
        }
    }

    /// Computes a reachability matrix for the current MP. A node is not considered to be
    /// reachable from itself unless it has a self-loop.
    ///
    /// Returns `reachability[i][j] := i~~>j`.
    ///
    /// The current implementation is O(n^4), but it probably doesn't matter.
    fn reachability(&self) -> Vec<Vec<Reachability>> {
        // reachability[i][j] := i~~>j... a node is not self-reachable unless it has a self-loop or
        // it can reach another node that can reach it.
        let mut reachability: Vec<Vec<_>> =
            vec![vec![Reachability::Unreachable; self.p.ncols()]; self.p.nrows()];

        // BFS
        for from in 0..self.p.nrows() {
            for to in 0..self.p.ncols() {
                if self.p[(from, to)] > f64::EPSILON {
                    reachability[from][to] = Reachability::Reachable;
                }
            }
        }
        for _ in 0..self.p.nrows() {
            for from in 0..self.p.nrows() {
                for mid in 0..self.p.ncols() {
                    for to in 0..self.p.ncols() {
                        if reachability[from][mid] == Reachability::Reachable
                            && reachability[mid][to] == Reachability::Reachable
                        {
                            reachability[from][to] = Reachability::Reachable;
                        }
                    }
                }
            }
        }

        reachability
    }

    /// Computes a list of transient nodes in the MP.
    fn transient_nodes(&self) -> Vec<usize> {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        enum Transience {
            Transient,
            Recurrent,
        }

        // Corner case: empty MP...
        if self.p.nrows() == 0 {
            return Vec::new();
        }

        // A node `a` is transient if there exists another node `b` such that `a~~>b` but not
        // `b~~>a`, i.e., we can get from `a` to `b` but not back.
        let reachability = self.reachability();
        let mut transience: Vec<_> = vec![Transience::Recurrent; self.p.nrows()];
        for node in 0..self.p.nrows() {
            for to in 0..self.p.ncols() {
                if reachability[node][to] == Reachability::Reachable
                    && reachability[to][node] == Reachability::Unreachable
                {
                    transience[node] = Transience::Transient;
                }
            }
        }

        // Return nodes identified as transient.
        transience
            .into_iter()
            .enumerate()
            .filter_map(|(i, transience)| matches!(transience, Transience::Transient).then_some(i))
            .collect()
    }

    /// Make the graph irreducbile. Get rid of transient nodes and connect unconnected components.
    /// This also nicely handles sink nodes.
    pub fn cleanup_mp(&mut self) {
        // Check for transient nodes and remove them.
        let transient_nodes = self.transient_nodes();
        self.remove_nodes(&transient_nodes);

        // We may have unconnected subgraphs, especially after culling edges or removing transient
        // nodes. Connect them by adding equally weighted edges from each connected component to
        // each of the others.
        let reachability = self.reachability();
        let mut connected_components = vec![0];
        for node in 0..self.p.nrows() {
            if connected_components
                .iter()
                .all(|cc| reachability[*cc][node] == Reachability::Unreachable)
            {
                connected_components.push(node);
            }
        }

        // Given that the original MP was probably not disconnected, we can guess that the edges
        // that were removed must have been unlikely overall. So when we add edges back to connect
        // the MP, let's not make the new edges likely.
        for &from in connected_components.iter() {
            for &to in connected_components.iter() {
                if from != to {
                    // will be normalized down... to < 0.1 / len(connected_components)
                    self.p[(from, to)] = 0.1;
                }
            }
            self.renormalize_row(from);
        }
    }

    /// Compute and return the stationary distribution of the MP.
    pub fn stationary_dist(&self) -> DVector<f64> {
        // Compute stationary distribution of markov process. For an aperiodic MP, the limiting
        // distribution L_a,b = lim_{n->inf} P(X_n = b | X_0 = a) will be equivalent to the stationary
        // distribution. Thus, we can approximate the stationary distribution by just raising the
        // probability transition matrix `p` to some large power.
        //
        // However, if the MP is periodic, then the limiting distribution will not exist. Instead, we
        // can find the periodic probabilities for a full cycle (deep in the future) and average them
        // together, since each state in the cycle is equally likely. This will give us the stationary
        // distribution.
        let limiting_approx = self.p.pow(1000);

        let period = {
            let mut period = 1; // start assuming aperiodic.
            let mut next = &limiting_approx * &self.p;
            loop {
                let diff = (&next - (&limiting_approx)).norm();
                if diff < 0.1 {
                    break;
                } else {
                    period += 1;
                    next = &next * &self.p; // take a step.
                }
            }
            period
        };

        let stationary: DMatrix<f64> = (0..period)
            .map(|i| {
                // using i+1 avoids dealing with i==0...
                (&limiting_approx) * self.p.pow(i + 1)
            })
            .sum::<DMatrix<_>>()
            / (period as f64);

        stationary.fixed_rows::<1>(0).transpose().into_owned()
    }

    pub fn p(&self) -> &DMatrix<f64> {
        &self.p
    }

    pub fn labels(&self) -> impl Iterator<Item = &CombinedGFPRegion> {
        self.labels.iter()
    }
}

#[cfg(test)]
#[test]
fn mp_test() {
    const EPSILON: f64 = 0.00001;

    {
        let mut mp = MarkovProcess {
            p: DMatrix::from_row_slice(3, 3, &[0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5]),
            labels: vec![
                CombinedGFPRegion {
                    order: 0,
                    flags: GFPFlags::None
                };
                3
            ],
        };

        mp.cleanup_mp();

        assert_eq!(mp.p.nrows(), 2);
        assert_eq!(mp.p.ncols(), 2);

        assert!(mp.p[(0, 0)] - 0.5 < EPSILON);
        assert!(mp.p[(1, 1)] - 0.5 < EPSILON);
        assert!(mp.p[(0, 1)] - 0.5 < EPSILON);
        assert!(mp.p[(1, 0)] - 0.5 < EPSILON);

        let s = mp.stationary_dist();
        assert!(s[0] - 0.5 < EPSILON);
        assert!(s[1] - 0.5 < EPSILON);
    }

    {
        let mut mp = MarkovProcess {
            p: DMatrix::from_row_slice(3, 3, &[0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            labels: vec![
                CombinedGFPRegion {
                    order: 0,
                    flags: GFPFlags::None
                };
                3
            ],
        };
        mp.cleanup_mp();

        assert_eq!(mp.p.nrows(), 2);
        assert_eq!(mp.p.ncols(), 2);

        assert!(mp.p[(0, 0)] - 0.909090909090909 < EPSILON);
        assert!(mp.p[(1, 1)] - 0.909090909090909 < EPSILON);
        assert!(mp.p[(0, 1)] - 0.090909090909090 < EPSILON);
        assert!(mp.p[(1, 0)] - 0.090909090909090 < EPSILON);

        let s = mp.stationary_dist();
        assert!(s[0] - 0.5 < EPSILON);
        assert!(s[1] - 0.5 < EPSILON);
    }
}

pub fn markov<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    simulated_flags: bool,
    simplify_mp: usize,
    print_p: bool,
) -> io::Result<()> {
    let flags = PairIterator::new(combine_and_clean_flags(
        reader,
        ignored_flags,
        simulated_flags,
    ));

    let mut mp = MarkovProcess::construct(flags);

    // Remove edges and nodes that are too unlikely. By default, this is just nodes that have a
    // probability too small to be represented, but the user can also set the threshold higher.
    mp.cull_unlikely(simplify_mp as f64 / MP_GRANULARITY);

    // Make the MP a bit easier to simulate.
    mp.cleanup_mp();

    /*
    {
        for node in 0..mp.p.nrows() {
            println!("Node {node} {}", mp.labels[node]);
            print!("Incoming:");
            for i in 0..mp.p.nrows() {
                if mp.p[(i, node)] > f64::EPSILON {
                    print!(" {i}({:0.3})", mp.p[(i, node)]);
                }
            }
            println!();
            print!("Outgoing:");
            for i in 0..mp.p.nrows() {
                if mp.p[(node, i)] > f64::EPSILON {
                    print!(" {i}({:0.3})", mp.p[(node, i)]);
                }
            }
            println!();
        }
    }
    */

    // Print the MP.
    for (i, fa) in mp.labels().enumerate() {
        print!("{} {:x}", fa.order, fa.flags as u64);
        for (j, _fb) in mp.labels().enumerate() {
            let prob = mp.p()[(i, j)];
            if prob >= 1.0 / MP_GRANULARITY {
                print!(" {j} {}", (prob * MP_GRANULARITY) as u64);
            }
        }
        print!(";")
    }
    println!();

    // Print the Stationary Dist.
    let stationary = mp.stationary_dist();
    print!("Stationary Distribution:");
    for (f, pi) in mp.labels().zip(stationary.iter()) {
        if *pi >= 1.0 / MP_GRANULARITY {
            print!(" {:x}:{}:{pi:0.3}", f.flags as u64, f.order);
        }
    }
    println!();

    Ok(())
}

pub fn empirical_dist<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> io::Result<()> {
    let flags = combine_and_clean_flags(reader, ignored_flags, simulated_flags);
    let mut stats = BTreeMap::new();
    let mut total = 0.0;

    // Iterate over contiguous physical memory regions with similar properties.
    for region in flags {
        let orders = stats
            .entry(region.flags)
            .or_insert_with(|| [0.0; MAX_ORDER as usize + 1]);
        orders[region.order as usize] += 1.0;
        total += 1.0;
    }

    // Print some stats about the different types of page usage.
    print!("Empirical Distribution:");
    for (flags, orders) in stats.into_iter() {
        for o in 0..orders.len() {
            if orders[o] / total >= 1.0 / MP_GRANULARITY {
                print!(" {:x}:{o}:{:0.3}", flags as u64, orders[o] / total);
            }
        }
    }

    Ok(())
}

pub fn type_dists<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> io::Result<()> {
    let flags = combine_and_clean_flags(reader, ignored_flags, simulated_flags);
    let mut stats = BTreeMap::new();

    // Iterate over contiguous physical memory regions with similar properties.
    for region in flags {
        let orders = stats
            .entry(region.flags)
            .or_insert_with(|| [0; MAX_ORDER as usize + 1]);
        orders[region.order as usize] += 1;
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
