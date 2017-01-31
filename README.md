atomic_ring_buffer
==================

_atomic ring buffer_ is a fixed-size multi-producer multi-consumer queue that works on
bare-metal systems. Under certain conditions, the queue is wait-free or lock-free.

See [documentation][docs] for details.

[docs]: https://docs.rs/atomic_ring_buffer/*/atomic_ring_buffer/struct.AtomicRingBuffer.html

Installation
------------

To use the _atomic ring buffer_ library in your project, add the following to `Cargo.toml`:

```toml
[dependencies]
atomic_ring_buffer = "1.0"
```

### Feature `std`

The `std` feature enables use `std::boxed::Box` and `std::vec::Vec` for the backing storage.

This feature is enabled by default.

### Feature `alloc`

The `alloc` feature enables use of `alloc::boxed::Box` for the backing storage.
This only works on nightly rustc.

### Feature `collections`

The `collections` feature enables use of `collections::vec::Vec` for the backing storage.
This only works on nightly rustc.

### Feature `const_fn`

The `const_fn` feature marks the `AtomicRingBuffer::new` function as constant, permitting
static initialization akin to:

```rust
static UART_BUFFER: AtomicRingBuffer = AtomicRingBuffer::new([u8; 32]);
```

This only works on nightly rustc.

License
-------

_atomic ring buffer_ is distributed under the terms of 0-clause BSD license.

See [LICENSE-0BSD](LICENSE-0BSD.txt) for details.
