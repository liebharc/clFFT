# Rust bindings for clFFT
Rust bindings for clFFT, a FFT library running on OpenCL devices. By default this library only compiles the bindings itself and therefore [prebuild binaries](https://github.com/clMathLibraries/clFFT/releases) are required in addition to the Rust bindings so that the library works. As an alternative the `build_all` feature flag can be used to build the clFFT library itself from the source using `cargo` and `cmake`.

## Related resources

- [Example](examples/example.rs)
- [Binding Documentation](https://liebharc.github.io/clFFT/bindings/clfft/)
- [Original README](CLFFT.md)
- [Original Documentation](http://clmathlibraries.github.io/clFFT/)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
clfft = "*"
```

and this to your crate root:

```rust
extern crate clfft;
```

## Build
In order to build only the bindings, run

```
cargo build
```

To also build `clFFT` itself you need to pass the feature flag `build_all`, e.g.

```
cargo build --features build_all,
```

This requires that `cmake` is installed on the system, refer to the [build page](https://github.com/clMathLibraries/clFFT/wiki/Build) for more details.