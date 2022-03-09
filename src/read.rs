//! Abstractions for reading kpageflags and producing a stream of flags.

use std::io::{self, BufRead};

use crate::{KPageFlags, KPF, KPF_SIZE};

/// Wrapper around a `BufRead` type that for the `/proc/kpageflags` file.
pub struct KPageFlagsReader<B: BufRead> {
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
pub struct KPageFlagsIterator<B: BufRead> {
    /// The reader we are reading from.
    reader: KPageFlagsReader<B>,

    /// Temporary buffer for data read but not consumed yet.
    buf: [KPageFlags; 1 << (21 - 3)],
    /// The number of valid flags in the buffer.
    nflags: usize,
    /// The index of the first valid, unconsumed flag in the buffer, if `nflags > 0`.
    idx: usize,

    ignored_flags: u64,
}

impl<B: BufRead> KPageFlagsIterator<B> {
    pub fn new(reader: KPageFlagsReader<B>, ignored_flags: &[KPF]) -> Self {
        KPageFlagsIterator {
            reader,
            buf: [KPageFlags::empty(); 1 << (21 - 3)],
            nflags: 0,
            idx: 0,
            ignored_flags: {
                let mut mask = 0;

                for f in ignored_flags.into_iter() {
                    mask |= 1 << (*f as u64);
                }

                mask
            },
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
        let mut item = self.buf[self.idx];

        item.clear(self.ignored_flags);

        self.nflags -= 1;
        self.idx += 1;

        Some(item)
    }
}
