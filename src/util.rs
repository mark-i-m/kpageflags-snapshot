//! Generic useful utilities.

use std::collections::VecDeque;

/// Types whose values are transitively combinable.
pub trait Combinable: Sized {
    /// Can `a` and `b` be combined? Note: this property is transitive.
    fn combinable(a: &Self, b: &Self) -> bool;

    /// Combine the given list of values into one value.
    fn combine(vals: &[Self]) -> Self;
}

/// Consumes a collection of iterators in lockstep. In each call to `next`, it calls
/// `next` on each of the sub-iterators and returns an array with the returned values.
pub struct LockstepIterator<I: Iterator> {
    iters: Vec<I>,
}

impl<I: Iterator> LockstepIterator<I> {
    pub fn new(iters: Vec<I>) -> Self {
        LockstepIterator { iters }
    }
}

impl<I: Iterator> Iterator for LockstepIterator<I> {
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

/// An iterator that returns a sliding window over `N` elements.
pub struct WindowIterator<I, const N: usize>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    iter: I,
    window: VecDeque<<I as Iterator>::Item>,
}

impl<I, const N: usize> WindowIterator<I, N>
where
    I: Iterator,
    <I as Iterator>::Item: Clone,
{
    pub fn new(mut iter: I) -> Self {
        assert!(N > 0);

        let mut window = VecDeque::with_capacity(N);
        for _ in 0..N {
            if let Some(v) = iter.next() {
                window.push_back(v);
            }
        }
        Self { iter, window }
    }
}

impl<I, const N: usize> Iterator for WindowIterator<I, N>
where
    I: Iterator,
    <I as Iterator>::Item: Clone + std::fmt::Debug,
{
    type Item = [<I as Iterator>::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.window.len() == N {
            let ret = self
                .window
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            self.window.pop_front();
            if let Some(next) = self.iter.next() {
                self.window.push_back(next);
            }
            Some(ret)
        } else {
            None
        }
    }
}

/// An iterator that returns (current, next) for all items in the stream.
pub struct PairIterator<I>
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

pub struct CombiningIterator<I: Iterator>(std::iter::Peekable<I>);

impl<I: Iterator> CombiningIterator<I> {
    pub fn new(iter: I) -> Self {
        CombiningIterator(iter.peekable())
    }
}

impl<I> Iterator for CombiningIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Combinable,
{
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(next) = self.0.next() else {
            return None;
        };

        let mut combined = vec![next];

        while let Some(next) = self.0.peek() {
            if <I as Iterator>::Item::combinable(&combined[0], next) {
                combined.push(self.0.next().unwrap());
            } else {
                break;
            }
        }

        Some(<I as Iterator>::Item::combine(&combined))
    }
}

/// Get the log (base 2) of `x`. `x` must be a power of two.
#[allow(dead_code)]
pub fn log2(x: u64) -> u64 {
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
