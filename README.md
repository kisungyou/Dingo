# Dingo

Dingo is a header-only C++ numerical library with an Armadillo-style API in the `dingo::` namespace, backed by Eigen.

## Highlights

- Dense containers: `Mat`, `Col`, `Row`
- 3D/container types: `Cube`, `Cell<T>`
- Core operations: arithmetic, broadcasting, reductions, reshape/vectorise, indexing helpers, elementwise math
- Linear algebra: `solve`, `inv`, `pinv`, `det`, `rank`, `cond`, `eig_*`, `svd`, `qr`, `chol`, `lu`, `log_det`, `kron`, `null`, `orth`
- Eigen interop: `as_eigen`, `from_eigen_mat`, `from_eigen_col`, `from_eigen_row`

## Quick Start

```cpp
#include <dingo>
```

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Scope (Current)

Implemented as an initial scaffold. Not yet included:

- File I/O APIs
- Sparse matrix APIs
- `kmeans` / Gaussian mixture model estimation

## Optional BLAS/LAPACK (vendored Eigen)

```bash
cmake -S . -B build \
  -DDINGO_USE_SYSTEM_EIGEN=OFF \
  -DDINGO_ENABLE_EIGEN_BLAS=ON \
  -DDINGO_ENABLE_EIGEN_LAPACK=ON
```

`DINGO_ENABLE_EIGEN_LAPACK=ON` requires `DINGO_ENABLE_EIGEN_BLAS=ON`.

## License

- Dingo: MIT (`LICENSE`)
- Third-party notices: `THIRD_PARTY_NOTICES.md`
