#![no_std]
#![cfg_attr(feature = "alloc", feature(alloc))]
#![cfg_attr(feature = "collections", feature(collections))]
#![cfg_attr(feature = "const_fn", feature(const_fn))]

//! See documentation for the [struct AtomicRingBuffer](struct.AtomicRingBuffer.html).

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;
#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::boxed::Box;

#[cfg(all(not(feature = "std"), feature = "collections"))]
extern crate collections;
#[cfg(all(not(feature = "std"), feature = "collections"))]
use collections::vec::Vec;

use core::usize;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::marker::PhantomData;
use core::cell::UnsafeCell;

/// A marker trait indicating that `as_mut()` and `as_ref()` always return
/// the same pointer.
///
/// When using `as_ref()` or `as_mut()`, it is generally reasonable to assume that:
///
///    * both return a pointer to the same data, and
///    * this pointer never changes.
///
/// These assumptions are vital when writing unsafe code.
///
/// However, it is legal (although likely counterproductive) to implement `as_ref()`
/// and `as_mut()` that violate this assumption, and when coupled with unsafe code
/// this results in breach of memory safety. Hence, this (unsafe to implement)
/// trait is used to indicate that the implementation is sane.
pub unsafe trait InvariantAsMut<T: ?Sized>: AsMut<T> + AsRef<T> {}

macro_rules! array_impl {
    () => ();
    ($n:expr) => (unsafe impl<T> InvariantAsMut<[T]> for [T; $n] {});
    ($n:expr, $( $r:expr ),*) => (array_impl!($n); array_impl!($( $r ),*););
}

unsafe impl<'a, T, U> InvariantAsMut<U> for &'a mut T
    where T: InvariantAsMut<U> + ?Sized, U: ?Sized {}

#[cfg(any(feature = "std", feature = "alloc"))]
unsafe impl<T: ?Sized> InvariantAsMut<T> for Box<T> {}

#[cfg(any(feature = "std", feature = "collections"))]
unsafe impl<T> InvariantAsMut<Vec<T>> for Vec<T> {}
#[cfg(any(feature = "std", feature = "collections"))]
unsafe impl<T> InvariantAsMut<[T]> for Vec<T> {}

unsafe impl<T> InvariantAsMut<[T]> for [T] {}

array_impl!(0,  1,  2,  3,   4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

/// The errors that may happen during an enqueue or a dequeue operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The ring buffer is full (during an enqueue operation) or empty
    /// (during a dequeue operation).
    Exhausted,
    /// Another operation of the same kind (enqueue or dequeue) is being performed.
    Locked
}

/// A fixed-size multi-producer multi-consumer queue that works on bare-metal systems.
///
/// An atomic ring buffer provides two basic operations: enqueue and dequeue, both acting
/// on a single element. Under certain restrictions, the queue is lock-free or wait-free.
///
/// Lock-freedom means that, when the whole program is run for sufficient time, at least
/// one thread of execution is making progress, although any individual thread may stall
/// indefinitely. This data structure is lock-free when every closure passed to `enqueue()`
/// or `dequeue()` terminates in a bounded time.
///
/// Wait-freedom means that any operation on the data structure terminates in a bounded time.
/// This data structure is wait-free with respect to a producer (consumer) when there is only
/// one producer (consumer).
///
/// # Modes of operation
///
/// This data structure provides different useful guarantees depending on how it's used.
/// It may be used as a locking multiple-producer multiple-consumer queue, lock-free
/// multiple-producer multiple-consumer queue, or a wait-free single-producer single-consumer
/// queue.
///
/// ## Locking queue
///
/// While the closure passed to `enqueue()` or `dequeue()` executes, there is a lock on
/// the producer or consumer side. No other thread of execution may enqueue or dequeue elements,
/// and the corresponding call will return `Err(Error::Locked)`.
///
/// Even when an operating system is present, the queue provides no functionality to suspend
/// and afterwards wake the current thread of execution. This can be added on top of the queue
/// using a building block such as [thread parking][park] or a [futex][].
///
/// [park]:  https://doc.rust-lang.org/nightly/std/thread/index.html#blocking-support-park-and-unpark
/// [futex]: https://en.wikipedia.org/wiki/Futex
///
/// ## Lock-free queue
///
/// It may seem odd to use locks for implementing a lock-free queue. However, when the lock
/// is always taken for a bounded amount of time, the lock-free property is preserved.
///
/// Defining the queue operations as passing a `&mut` pointer has several benefits: there is
/// no requirement for a `T: Copy` bound; exactly one copy will be performed; it is possible
/// to read or write only a part of the element.
///
/// Another benefit is implementation simplicity. This queue implementation uses contiguous
/// storage and two indexes to manage itself. In order to ensure atomic updates, a 'true'
/// lock-free implementation would have to use a linked list, which inflicts excessive overhead
/// for small elements (e.g. `u8`) and makes it impossible to initialize a `static` with
/// a new queue.
///
/// To aid the lock-free mode of operation, the `enqueue_spin()` and `dequeue_spin()` methods
/// can be used, which retry the operation every time one fails with an `Error::Locked`.
///
/// ## Wait-free queue
///
/// Lock-freedom is not always enough. In a real-time system, not only the entire program must
/// make progress, but also certain threads of execution must make progress quickly enough.
///
/// When there is only one producer (only one consumer), the enqueue (dequeue) operation is
/// guaranteed to terminate in bounded time, and never return `Err(Error::Locked)`.
/// If there is both only a single producer and a single consumer, the queue is fully wait-free.
///
/// A partially wait-free queue is still useful. For example, if only an interrupt handler bound
/// to a single core produces elements, but multiple threads consume them, the interrupt handler
/// is still guaranteed to terminate in bounded time.
///
/// # Panic safety
///
/// The queue maintains memory safety even if an enqueue or a dequeue operation panics.
/// However, it will remain locked forever.
pub struct AtomicRingBuffer<T, U: InvariantAsMut<[T]>> {
    phantom: PhantomData<T>,
    /// Reader pointer, modulo `storage.len() * 2`.
    /// The next dequeue operation will read from storage[reader].
    reader:  AtomicUsize,
    /// Writer pointer, modulo `storage.len() * 2`.
    /// The next enqueue operation will write to storage[writer].
    writer:  AtomicUsize,
    /// Underlying storage. Wrapped in an UnsafeCell, since a dequene and an enqueue operation
    /// may return a &mut pointer to different parts of the storage.
    storage: UnsafeCell<U>
}

unsafe impl<T, U: InvariantAsMut<[T]>> Sync for AtomicRingBuffer<T, U> {}

// To avoid the need for a separate empty/non-empty flag (which couldn't really be integrated
// with usize atomic operations, since both the reader and the writer may need to set it),
// as well as to avoid losing one element of the buffer (which is fine if the element is an u8
// but less fine when it's a [u8; 1500] network packet buffer), we use a clever trick.
//
// Specifically, we treat both reader and writer pointers as modulo capacity when accessing
// the storage, but modulo capacity * 2 when advancing or comparing them. As such, if, say,
// we have the capacity 5, r=3 w=3 means an empty buffer, and r=3 w=8 means a full buffer.
//
// This is an improved variant of the technique presented in [JuhoSnellman][], which doesn't
// have the drawback of requiring power-of-2 buffers (which is not currently possible to enforce
// when using `const fn`).
//
// [JuhoSnellman]: https://www.snellman.net/blog/archive/2016-12-13-ring-buffers/.

/// An `usize` value with only the MSB set.
const LOCK_BIT: usize = (core::usize::MAX - 1) / 2 + 1;

impl<T, U: InvariantAsMut<[T]>> AtomicRingBuffer<T, U> {
    #[cfg(not(feature = "const_fn"))]
    /// Create a ring buffer.
    pub fn new(storage: U) -> AtomicRingBuffer<T, U> {
        AtomicRingBuffer {
            storage: UnsafeCell::new(storage),
            reader:  AtomicUsize::new(0),
            writer:  AtomicUsize::new(0),
            phantom: PhantomData
        }
    }

    #[cfg(feature = "const_fn")]
    /// Create a ring buffer.
    pub const fn new(storage: U) -> AtomicRingBuffer<T, U> {
        AtomicRingBuffer {
            storage: UnsafeCell::new(storage),
            reader:  AtomicUsize::new(0),
            writer:  AtomicUsize::new(0),
            phantom: PhantomData
        }
    }

    /// Enqueue an element.
    ///
    /// This function returns `Err(Error::Exhausted)` if the queue is full,
    /// or `Err(Error::Locked)` if another enqueue operation is in progress.
    #[inline]
    pub fn enqueue<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> Result<R, Error> {
        let storage  = unsafe { (*self.storage.get()).as_ref() };
        let capacity = storage.len();

        loop {
            let writer = self.writer.load(Ordering::SeqCst);
            // If the writer pointer changes after this point, we'll detect it when doing CAS.

            if writer & LOCK_BIT != 0 {
                // Someone is already enqueueing an element.
                return Err(Error::Locked)
            }

            if self.writer.compare_and_swap(writer, writer | LOCK_BIT,
                                            Ordering::SeqCst) != writer {
                // Someone else has enqueued an element, start over.
                continue
            }

            let reader = self.reader.load(Ordering::SeqCst);
            // If the reader pointer advances after this point, we may (wrongly) determine
            // that the queue is full, when it is not. This is conservative and fine.

            if (writer + capacity) % (2 * capacity) == (reader & !LOCK_BIT) {
                // The queue is full.
                self.writer.store(writer, Ordering::SeqCst);
                return Err(Error::Exhausted)
            }

            // At this point, we have exclusive access over storage[writer].
            let ptr = unsafe { storage.as_ptr().offset((writer % capacity) as isize) };
            let result = f(unsafe { &mut *(ptr as *mut T) });

            // Advance the write index, and release the lock, in the same operation.
            let next_writer = (writer + 1) % (capacity * 2);
            self.writer.store(next_writer, Ordering::SeqCst);

            return Ok(result)
        }
    }

    /// Dequeue an element.
    ///
    /// This function returns `Err(Error::Exhausted)` if the queue is empty,
    /// or `Err(Error::Locked)` if another dequeue operation is in progress.
    #[inline]
    pub fn dequeue<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> Result<R, Error> {
        let storage  = unsafe { (*self.storage.get()).as_ref() };
        let capacity = storage.len();

        loop {
            let reader = self.reader.load(Ordering::SeqCst);
            // If the reader pointer changes after this point, we'll detect it when doing CAS.

            if reader & LOCK_BIT != 0 {
                // Someone is already dequeueing an element.
                return Err(Error::Locked)
            }

            if self.reader.compare_and_swap(reader, reader | LOCK_BIT,
                                            Ordering::SeqCst) != reader {
                // Someone else has enqueued an element, start over.
                continue
            }

            let writer = self.writer.load(Ordering::SeqCst);
            // If the writer pointer advances after this point, we may (wrongly) determine
            // that the queue is empty, when it is not. This is conservative and fine.

            if reader == (writer & !LOCK_BIT) {
                // The ring buffer is empty.
                self.reader.store(reader, Ordering::SeqCst);
                return Err(Error::Exhausted)
            }

            // At this point, we have exclusive access over storage[reader].
            let ptr = unsafe { storage.as_ptr().offset((reader % capacity) as isize) };
            let result = f(unsafe { &mut *(ptr as *mut T) });

            // Advance the read index, and release the lock, in the same operation.
            self.reader.store((reader + 1) % (capacity * 2), Ordering::SeqCst);

            return Ok(result)
        }
    }

    /// Enqueue an element.
    ///
    /// This function returns `Err(())` if the queue is full, and retries the operation
    /// if another enqueue operation is in progress.
    #[inline]
    pub fn enqueue_spin<F: FnMut(&mut T) -> R, R>(&self, mut f: F) -> Result<R, ()> {
        loop {
            match self.enqueue(&mut f) {
                Ok(result) => return Ok(result),
                Err(Error::Locked) => continue,
                Err(Error::Exhausted) => return Err(())
            }
        }
    }

    /// Dequeue an element.
    ///
    /// This function returns `Err(())` if the queue is empty, and retries the operation
    /// if another dequeue operation is in progress.
    #[inline]
    pub fn dequeue_spin<F: FnMut(&mut T) -> R, R>(&self, mut f: F) -> Result<R, ()> {
        loop {
            match self.dequeue(&mut f) {
                Ok(result) => return Ok(result),
                Err(Error::Locked) => continue,
                Err(Error::Exhausted) => return Err(())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::vec::Vec;
    use std::thread;
    use std::sync::{Arc, Mutex};
    use super::*;

    #[cfg(feature = "const_fn")]
    static TEST_CONST_FN: AtomicRingBuffer<u8, [u8; 16]> = AtomicRingBuffer::new([0; 16]);

    #[test]
    fn test_single_elem() {
        let queue = AtomicRingBuffer::new([0u8; 1]);
        assert_eq!(queue.dequeue(|x| *x), Err(Error::Exhausted));
        assert_eq!(queue.enqueue(|x| *x = 1), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 1), Err(Error::Exhausted));
        assert_eq!(queue.dequeue(|x| *x), Ok(1));
    }

    #[test]
    fn test_four_elems() {
        let queue = AtomicRingBuffer::new([0u8; 4]);
        assert_eq!(queue.enqueue(|x| *x = 1), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 2), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 3), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 4), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 5), Err(Error::Exhausted));
        assert_eq!(queue.dequeue(|x| *x), Ok(1));
        assert_eq!(queue.dequeue(|x| *x), Ok(2));
        assert_eq!(queue.enqueue(|x| *x = 5), Ok(()));
        assert_eq!(queue.enqueue(|x| *x = 6), Ok(()));
        assert_eq!(queue.dequeue(|x| *x), Ok(3));
        assert_eq!(queue.dequeue(|x| *x), Ok(4));
        assert_eq!(queue.dequeue(|x| *x), Ok(5));
        assert_eq!(queue.dequeue(|x| *x), Ok(6));
        assert_eq!(queue.dequeue(|x| *x), Err(Error::Exhausted));
    }

    #[test]
    fn test_locking() {
        let queue = AtomicRingBuffer::new([0u8; 4]);
        // enqueue while enqueueing
        queue.enqueue(|_| {
            assert_eq!(queue.enqueue(|x| *x = 1), Err(Error::Locked));
        }).unwrap();
        // dequeue while dequeueing
        assert_eq!(queue.enqueue(|x| *x = 1), Ok(()));
        queue.dequeue(|_| {
            assert_eq!(queue.dequeue(|x| *x), Err(Error::Locked));
        }).unwrap();
    }

    const BUF_SIZE: usize = 10;
    const COUNT_TO: usize = 1_000_000;

    #[test]
    fn test_stress_mpsc() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Elem {
            X,
            A(usize),
            B(usize)
        }

        let queue = Arc::new(AtomicRingBuffer::new([Elem::X; BUF_SIZE]));

        macro_rules! producer {
            ($queue:ident, $kind:ident) => ({
                let queue = $queue.clone();
                thread::spawn(move || {
                    for counter in 0..COUNT_TO {
                        loop {
                            match queue.enqueue_spin(|x| *x = Elem::$kind(counter)) {
                                Ok(()) => break,
                                Err(()) => continue
                            }
                        }
                    }
                })
            })
        }

        let a_producer = producer!(queue, A);
        let b_producer = producer!(queue, B);

        let consumer = thread::spawn(move || {
            let mut a_counter = 0;
            let mut b_counter = 0;
            loop {
                match queue.dequeue_spin(|x| *x) {
                    Ok(Elem::A(n)) if a_counter == n => a_counter += 1,
                    Ok(Elem::B(n)) if b_counter == n => b_counter += 1,
                    Ok(elem) => panic!("unexpected {:?}; a_c={}, b_c={}",
                                       elem, a_counter, b_counter),
                    Err(()) if a_counter == COUNT_TO && b_counter == COUNT_TO => break,
                    Err(()) => ()
                }
            }
        });

        a_producer.join().unwrap();
        b_producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn test_stress_spmc() {
        let queue = Arc::new(AtomicRingBuffer::new([0usize; BUF_SIZE]));
        let results = Arc::new(Mutex::new(Vec::new()));

        macro_rules! consumer {
            ($queue:ident, $results:ident) => ({
                let queue = $queue.clone();
                let results = $results.clone();
                thread::spawn(move || {
                    let mut counter = 0;
                    loop {
                        match queue.dequeue_spin(|x| *x) {
                            Ok(x) if x == 0 => break,
                            Ok(x) if x > counter => {
                                results.lock().unwrap().push(x);
                                counter = x;
                            }
                            Ok(_) => panic!("out-of-order counter values"),
                            Err(()) => continue
                        }
                    }
                })
            })
        }

        let a_consumer = consumer!(queue, results);
        let b_consumer = consumer!(queue, results);

        let producer = thread::spawn(move || {
            for counter in (1..COUNT_TO).chain([0, 0].iter().map(|x| *x)) {
                loop {
                    match queue.enqueue_spin(|x| *x = counter) {
                        Ok(()) => break,
                        Err(()) => continue
                    }
                }
            }
        });

        producer.join().unwrap();
        a_consumer.join().unwrap();
        b_consumer.join().unwrap();

        let mut results = results.lock().unwrap();
        assert_eq!(results.len(), COUNT_TO - 1);

        results.sort();
        for counter in 0..(COUNT_TO - 1) {
            assert_eq!(results[counter], counter + 1);
        }
    }
}
