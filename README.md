![Static Badge](https://img.shields.io/badge/github-einops--cpp-red?style=flat-square&logo=github)
![Static Badge](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)
![Static Badge](https://img.shields.io/badge/release-v0.1a-blue?style=flat-square&color=blue)
![Static Badge](https://img.shields.io/badge/build-passing-green?style=flat-square)

# einops-cpp - C++ port of einops for libtorch backend

**einops-cpp** is a **C++17** compatible **header-only** library that implement the [einops](https://github.com/arogozhnikov/einops) python project developed by **Alex Rogozhnikov** and elegantly summarized: *"flexible and powerful tensor operations for readable and reliable code"*. 

## Installation

No installation needed, this is a **zero-dependencies** (except **libtorch**) library.
Just put the `'include'` directory on your compiler path and add the following line somewhere in you code:  

```cpp
#include <einops.hpp>
```

## Build

**No build is needed** for the library. However, a very basic cmake file is ready to build the test project. It partially follows the different tests of the python project. Build successfull with MSVC 17.7.4 & LLVM-Clang (VS 2022 and GCC 11.3 on Ubuntu 22-04 (WSL2).

## Usage

```cpp
#include <einops.hpp>
using namespace einops;

auto x = torch::arange({ 10 * 20 * 30 * 40 }).reshape({ 10, 20, 30, 40 });

// here it is an example of max-pooling with einops
auto y = reduce(x, "b c (h h1) (w w1) -> b c h w", "max", axis("h1", 2), axis("w1", 2));
```
  
> [!IMPORTANT]   
> `axis(key,value)` is an helper to simulate the syntax of the axes lengths, in python : `(..., h1=2, w1=2)`

## Documentation

All the following methods in the public C++ API are documented. For a better understanding, take a look at the test project which contains examples for each of the public methods. You can also check the original python project documentation [einops](https://einops.rocks/).

## Status

- [x] Follow the code of the python release `v0.7.0` [release](https://github.com/arogozhnikov/einops/releases/tag/v0.7.0)
- [x] Fully implements the `reduce()`, `rearrange()`, `repeat()`, `einsum()` and `parse_shape()` methods.
- [ ] Finalize the code of the `Rearrange`, `Reduce` and `EinMix` layers (aka `torch::Module`)
- [ ] Need to benchmark the LRU cache in few internal methods
- [ ] Should optimize the code where possible (limit potential overhead)

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Future versions & refactoring
The current code is a direct port of the **Python** code, libtorch is relatively easy to be adapted in **C++**. A future version will not be directly based on the python code, but rewritten for support another backends (e.g. Tensorflow, XTensor and more).  

**Tensorflow** C++ API flow is not adapted to the current implementation (need scope, graph flow and session) and need a concatenation of the operations in a session at '`_apply_recipe`' level for a better optimization of the code.  

**XTensor** need type specialization by template for dynamic tensor instanciation (aka `xt::xarray`), so the current abstraction of the Backend base class need to be rewrite.

The compile-time limitations of some potential future backend support, think **Fastor** ([here](https://github.com/romeric/Fastor)), force to rethink the whole architecture to adapt both runtime and compile-time process.