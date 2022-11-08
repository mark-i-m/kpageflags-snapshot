//! Logic for processing kpageflags.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
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

use crate::{util::*, Args};

/// The `MAX_ORDER` for Linux 5.17 (and a lot of older versions).
pub const MAX_ORDER: u64 = 10;

/// The number of 4KB pages in a `MAX_ORDER`-sized chunk of memory.
pub const MAX_ORDER_PAGES: u64 = 1 << MAX_ORDER;

/// The  granularity with which probabilities are expressed in MPs.
pub const MP_GRANULARITY: f64 = 1E3;

/// The number of steps of history to use.
pub const MP_HISTORY_LEN: usize = 1;

#[derive(Copy, Clone, Debug)]
pub struct CombinedPageFlags<K: Flaggy> {
    /// Starting PFN (inclusive).
    pub start: u64,
    /// Ending PFN (exlusive).
    pub end: u64,

    /// Flags of the indicated region.
    pub flags: KPageFlags<K>,
}

impl<K: Flaggy> CombinedPageFlags<K> {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
}

impl<K: Flaggy> Combinable for CombinedPageFlags<K> {
    fn combinable(a: &[Self], b: &Self) -> bool {
        let total_len: u64 = a.iter().map(|r| r.len()).sum();
        let ty = a[0].flags;
        (total_len + b.len()) <= MAX_ORDER_PAGES && KPageFlags::can_combine(ty, b.flags)
    }

    fn combine(vals: &[Self]) -> Self {
        CombinedPageFlags {
            start: vals.iter().map(|cpf| cpf.start).min().unwrap(),
            end: vals.iter().map(|cpf| cpf.end).max().unwrap(),
            flags: vals
                .iter()
                .map(|cpf| cpf.flags)
                .reduce(|a, b| a | b)
                .unwrap(),
        }
    }
}

pub fn map_and_summary<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    args: &Args,
) -> io::Result<()> {
    let flags = clean_flags(reader, ignored_flags, args.simulated_flags);
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
                let size = region.len() * 4;
                println!(
                    "{:010X}-{:010X} {:8}KB {}",
                    region.start, region.end, size, region.flags,
                );
            }
        }

        let (total, stats) = stats
            .entry(region.flags)
            .or_insert_with(|| (0, Histogram::<u64>::new(5).unwrap()));

        *total += region.len();
        stats.record(region.len()).unwrap();
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
    /// The size of the allocation in number of 4KB pages.
    pub npages: u64,

    /// GFP for the given region.
    pub flags: GFPFlags,
}

impl Ord for CombinedGFPRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let flagscmp = Ord::cmp(&self.flags, &other.flags);
        let npagescmp = Ord::cmp(&self.npages, &other.npages);

        if flagscmp == Ordering::Equal {
            npagescmp
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
        self.npages.hash(state);
        self.flags.hash(state);
    }
}

impl Display for CombinedGFPRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.flags, self.npages)
    }
}

impl MarkovLabel for CombinedGFPRegion {
    fn ty(&self) -> u64 {
        self.flags as u64
    }
    fn npages(&self) -> u64 {
        self.npages
    }
}

impl Combinable for CombinedGFPRegion {
    fn combinable(a: &[Self], b: &Self) -> bool {
        let total_len: u64 = a.iter().map(|r| r.npages).sum();
        a[0].flags == b.flags && (total_len + b.npages) <= MAX_ORDER_PAGES
    }

    fn combine(vals: &[Self]) -> Self {
        CombinedGFPRegion {
            npages: vals.iter().map(|cgfpr| cgfpr.npages).sum(),
            flags: vals[0].flags,
        }
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
#[allow(dead_code)]
fn break_into_pow_of_2_regions(size: u64) -> Vec<u64> {
    // The set of orders we want is already represented in the binary representation of `size`.

    const MAX_ORDER_MAX_LO: u64 = MAX_ORDER_PAGES - 1;
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

fn clean_flags<K: Flaggy>(
    reader: FileReadableReader<impl Read, KPageFlags<K>>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> impl Iterator<Item = CombinedPageFlags<K>> {
    CombiningIterator::new(
        KPageFlagsIterator::new(reader, ignored_flags)
            .enumerate()
            .map(|(i, flags)| CombinedPageFlags {
                start: i as u64,
                end: i as u64 + 1,
                flags,
            }),
    )
    .map(move |mut region| {
        if simulated_flags && region.flags.all(K::OWNERPRIVATE1 | K::RESERVED) {
            region.flags.clear(K::OWNERPRIVATE1 | K::RESERVED);

            // Anon THP pages.
            if region.flags.all(K::PRIVATE | K::PRIVATE2) {
                region.flags.clear(K::PRIVATE | K::PRIVATE2);
                region.flags |= KPageFlags::from(K::ANON | K::THP);
            }
            // Anon pages.
            else if region.flags.all(K::PRIVATE) {
                region.flags.clear(K::PRIVATE);
                region.flags |= KPageFlags::from(K::ANON);
            }
            // File pages.
            else if region.flags.all(K::LRU) {
                // Nothing to do...
            }
            // Private 2 without Private Cannot happen!
            else if region.flags.all(K::PRIVATE2) {
                unreachable!();
            }
            // Pinned pages.
            else {
                region.flags |= KPageFlags::from(K::SLAB);
            }
        }

        region
    })
}

fn clean_and_combine_flags<K: Flaggy>(
    reader: FileReadableReader<impl Read, KPageFlags<K>>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> impl Iterator<Item = CombinedGFPRegion> {
    CombiningIterator::new(
        clean_flags(reader, ignored_flags, simulated_flags)
            .filter(|region| !region.flags.any(K::NOPAGE))
            .filter(|region| !region.flags.any(K::RESERVED))
            .map(|region| CombinedGFPRegion {
                npages: region.len(),
                flags: kpf_to_abstract_flags(region.flags),
            }),
        //.flat_map(|combined| {
        //    break_into_pow_of_2_regions(combined.len())
        //        .into_iter()
        //        .map(move |order| CombinedGFPRegion {
        //            order,
        //            flags: kpf_to_abstract_flags(combined.flags),
        //        })
        //})
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Reachability {
    Reachable,
    Unreachable,
}

trait MarkovLabel {
    fn ty(&self) -> u64;
    fn npages(&self) -> u64;
}

impl<L: MarkovLabel, const N: usize> MarkovLabel for [L; N] {
    fn ty(&self) -> u64 {
        if N == 0 {
            panic!();
        }
        self.last().unwrap().ty()
    }
    fn npages(&self) -> u64 {
        if N == 0 {
            panic!();
        }
        self.last().unwrap().npages()
    }
}

/// Represents a Markov Process with the ability to cull unlikely states and check irreducibility.
struct MarkovProcess<L> {
    // The main representation of the process is via the transition probability matrix. The matrix
    // shows the weight of each transition. We also keep a list of labels corresponding to the
    // different indices of the matrix. Both the label list and the matrix are in the same order.
    /// Transition probability matrix `p`: `p[i,j]` is the probability of going from `i` to `j` in
    /// one step. `labels[i]` is the label of `i`.
    p: DMatrix<f64>,

    /// Node labels.
    labels: Vec<L>,
}

impl<L> MarkovProcess<L>
where
    L: std::fmt::Debug + Clone + Ord + MarkovLabel,
{
    /// Construct a MP from the given iterator.
    pub fn construct(flags: impl Iterator<Item = (L, L)>) -> Self {
        // graph[a][b] = number of edges from a -> b in the graph.
        let mut graph = BTreeMap::new();
        for (fa, fb) in flags {
            // Add the edge...
            *graph
                .entry(fa.clone())
                .or_insert_with(BTreeMap::new)
                .entry(fb.clone())
                .or_insert(0.0) += 1.0;

            // And make sure both nodes are in the graph.
            graph.entry(fb.clone()).or_insert_with(BTreeMap::new);

            //if fa == fb {
            //    println!("{fa:?} -> {fb:?}");
            //}
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
        //let mut nremoved = 0;

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

            //nremoved += culled_nodes.len();

            // If no edges were removed, we reached a fixed point.
            if culled_nodes.is_empty() {
                //println!("Culled: {nremoved}");
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
    fn reachability(&self) -> Vec<Vec<Reachability>> {
        let mut reachable_sets = Vec::with_capacity(self.p.nrows());
        for from in 0..self.p.nrows() {
            reachable_sets.push(BTreeSet::new());
            for to in 0..self.p.ncols() {
                if self.p[(from, to)] > f64::EPSILON {
                    reachable_sets[from].insert(to);
                }
            }
        }

        loop {
            let mut changed = false;

            for from in 0..self.p.nrows() {
                let mut new_reachable = BTreeSet::new();
                for &to in reachable_sets[from].iter() {
                    new_reachable.extend(&reachable_sets[to] - &reachable_sets[from]);
                }
                if !new_reachable.is_empty() {
                    changed = true;
                }
                reachable_sets[from].append(&mut new_reachable); // empties new_reachable
            }

            if !changed {
                break;
            }
        }

        // reachability[i][j] := i~~>j... a node is not self-reachable unless it has a self-loop or
        // it can reach another node that can reach it.
        let mut reachability: Vec<Vec<_>> =
            vec![vec![Reachability::Unreachable; self.p.ncols()]; self.p.nrows()];
        for from in 0..self.p.nrows() {
            for to in 0..self.p.ncols() {
                if reachable_sets[from].contains(&to) {
                    reachability[from][to] = Reachability::Reachable;
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

        //println!("Connected Components: {}", connected_components.len());

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
                if diff.is_nan() {
                    panic!("diff is NaN");
                } else if diff < 0.1 {
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

    pub fn labels(&self) -> impl Iterator<Item = &L> {
        self.labels.iter()
    }

    #[allow(dead_code)]
    pub fn remove_similar_neighborships(&mut self) {
        let similar_labels = {
            let mut similar_labels = self
                .labels
                .iter()
                .map(|label| label.ty())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|ty| (ty, Vec::new()))
                .collect::<BTreeMap<_, _>>();

            for (i, label) in self.labels.iter().enumerate() {
                similar_labels.get_mut(&label.ty()).unwrap().push(i);
            }

            similar_labels
        };

        let mut completely_removed = Vec::new();
        for (_ty, labelsi) in similar_labels.iter() {
            for &labela in labelsi.iter() {
                for &labelb in labelsi.iter() {
                    if self.labels[labela].npages() < MAX_ORDER_PAGES
                        && self.labels[labelb].npages() < MAX_ORDER_PAGES
                    {
                        self.p[(labela, labelb)] = 0.0;
                    }
                }
                if self.p.row(labela).sum() < f64::EPSILON {
                    completely_removed.push(labela);
                }
            }
        }

        self.remove_nodes(&completely_removed);

        for fa in 0..self.p.nrows() {
            self.renormalize_row(fa);
        }
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
                    npages: 1,
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
                    npages: 1,
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

fn mp_label_fmt(label: &[CombinedGFPRegion]) -> String {
    let mut s = String::new();

    for (i, f) in label.iter().enumerate() {
        if i > 0 {
            s.push('|');
        }
        s.push_str(&format!("{}|{:x}", f.npages, f.flags as u64));
    }

    s
}

pub fn markov<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    simulated_flags: bool,
    simplify_mp: usize,
    print_p: bool,
) -> io::Result<()> {
    let flags = PairIterator::new(WindowIterator::<_, MP_HISTORY_LEN>::new(
        clean_and_combine_flags(reader, ignored_flags, simulated_flags),
    ));

    let mut mp = MarkovProcess::construct(flags);

    // Remove edges and nodes that are too unlikely. By default, this is just nodes that have a
    // probability too small to be represented, but the user can also set the threshold higher.
    mp.cull_unlikely(simplify_mp as f64 / MP_GRANULARITY);

    // Remove transitions to similar states.
    //mp.remove_similar_neighborships();

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

    // Print P.
    if print_p {
        for (i, _fa) in mp.labels().enumerate() {
            for (j, _fb) in mp.labels().enumerate() {
                print!("{} ", mp.p()[(i, j)]);
            }
            println!();
        }
    }

    // Print the MP.
    for (i, fa) in mp.labels().enumerate() {
        print!("{}", mp_label_fmt(fa));
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
    print!("Stationary Distribution:");
    let stationary = mp.stationary_dist();
    for (f, pi) in mp.labels().zip(stationary.iter()) {
        if *pi >= 1.0 / MP_GRANULARITY {
            print!(" {}:{pi:0.3}", mp_label_fmt(f));
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
    let flags = WindowIterator::<_, MP_HISTORY_LEN>::new(clean_and_combine_flags(
        reader,
        ignored_flags,
        simulated_flags,
    ));
    let mut stats = BTreeMap::new();
    let mut total = 0.0;

    for region in flags {
        *stats.entry(region).or_insert(0.0) += 1.0;
        total += 1.0;
    }

    print!("Empirical Distribution:");
    for (region, count) in stats.into_iter() {
        if count / total >= 1.0 / MP_GRANULARITY {
            print!(" {}:{:0.3}", mp_label_fmt(&region), count / total);
        }
    }

    Ok(())
}

pub fn type_dists<R: Read, K: Flaggy>(
    reader: KPageFlagsReader<R, K>,
    ignored_flags: &[K],
    simulated_flags: bool,
) -> io::Result<()> {
    let flags = clean_and_combine_flags(reader, ignored_flags, simulated_flags);
    let mut stats = BTreeMap::new();

    // Iterate over contiguous physical memory regions with similar properties.
    let mut sizes = BTreeSet::new();
    for region in flags {
        let npages = stats.entry(region.flags).or_insert_with(BTreeMap::new);
        *npages.entry(region.npages as usize).or_insert(0) += 1;
        sizes.insert(region.npages as usize);
    }

    // Print some stats about the different types of page usage.
    println!("Region type/size counts\n==========");
    let mut subtotals = Vec::new();
    for (flags, npages) in stats.iter() {
        for o in sizes.iter() {
            print!("{:8} ", npages.get(o).unwrap_or(&0));
        }
        let sub = npages.values().sum::<usize>();
        println!("{flags:4} {sub:8}");
        subtotals.push(sub);
    }
    println!(
        "Total region count: {}",
        subtotals.into_iter().sum::<usize>()
    );

    //println!("\nMemory Amounts (pages)\n==========");
    //let mut subtotals = Vec::new();
    //for (flags, orders) in stats.into_iter() {
    //    for o in 0..orders.len() {
    //        print!("{:8} ", orders[o] << o);
    //    }
    //    let sub = orders
    //        .iter()
    //        .enumerate()
    //        .map(|(o, n)| n << o)
    //        .sum::<usize>();
    //    println!("{flags:4} {sub:8}");
    //    subtotals.push(sub);
    //}
    //println!("Total memory: {}", subtotals.into_iter().sum::<usize>());

    Ok(())
}

pub fn compare_snapshots<K: Flaggy>(ignored_flags: &[K], args: &Args) -> io::Result<()> {
    let snapshot_iterators = LockstepIterator::new(
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
