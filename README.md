# Dingo

Dingo is a header-only C++ numerical library that provides an Armadillo-style API in the `dingo::` namespace while using Eigen internally.

## Status

Initial scaffold with:

- Dense containers: `dingo::Mat`, `dingo::Col`, `dingo::Row`
- 3D array container: `dingo::Cube`
- Array list container: `dingo::Cell<T>`
- Core routines: constructors, fills, arithmetic, broadcasting (`Mat` with `Col`/`Row`), reductions (including dim-wise), reshape/vectorise/join, diag/tril/triu, elementwise transforms, sorting/indexing helpers (`sort`, `unique`, `find`, `elem`), cube helpers (`squeeze`, `permute`)
- Linear algebra: `solve`, `inv`, `pinv`, `det`, `rank`, `cond`, `eig_sym`, `eig_gen`, `svd`, `qr`, `chol`, `lu`, `log_det`, `kron`, `null`, `orth`
- Interop helpers for Eigen-backed workflows (`as_eigen`, `from_eigen_mat`, `from_eigen_col`, `from_eigen_row`)
- Header include shim for `#include <dingo>`

Not included in v1 scaffold:

- File I/O APIs
- Sparse matrix APIs
- `kmeans` and Gaussian mixture model estimation

## Build and Tests

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If `-DDINGO_USE_SYSTEM_EIGEN=OFF` (default), provide a vendored Eigen snapshot under `third_party/eigen/Eigen`.

### Optional bundled BLAS/LAPACK (from vendored Eigen)

If you vendor Eigen's `blas/` and `lapack/` directories, you can build them with Dingo:

```bash
cmake -S . -B build \
  -DDINGO_USE_SYSTEM_EIGEN=OFF \
  -DDINGO_ENABLE_EIGEN_BLAS=ON \
  -DDINGO_ENABLE_EIGEN_LAPACK=ON
```

Notes:

- `DINGO_ENABLE_EIGEN_LAPACK=ON` requires `DINGO_ENABLE_EIGEN_BLAS=ON`.
- This builds and links bundled static BLAS/LAPACK libraries with Dingo.
- To force Eigen BLAS/LAPACKE backend macros in consumer translation units, enable:
  - `-DDINGO_USE_EIGEN_BLAS_BACKEND=ON`
  - `-DDINGO_USE_EIGEN_LAPACKE_BACKEND=ON` (requires LAPACKE availability)
- Backend macro options are OFF by default because behavior can be toolchain/platform sensitive.
- `DINGO_EIGEN_MPL2_ONLY=ON` (default) defines `EIGEN_MPL2_ONLY` to prevent accidental use of non-MPL2 Eigen components.

## Include

```cpp
#include <dingo>
```

or

```cpp
#include <dingo/dingo.hpp>
```

MATLAB-style indexing examples:

```cpp
dingo::mat A{{1,2,3},{4,5,6},{7,8,9}};
auto B = A(dingo::range(0,1), dingo::range(1,2)); // block read
A(dingo::all_idx, dingo::range(0,0)) = 0.0;       // block assignment

dingo::mat mask{{0,1,0},{1,0,1},{0,0,0}};
auto v = A(mask);  // selected values as Nx1 matrix
A(mask) = -1.0;    // masked assignment
```

## Benchmark (Dingo vs raw Eigen)

Build and run:

```bash
/Applications/CMake.app/Contents/bin/cmake -S . -B build \
  -DDINGO_USE_SYSTEM_EIGEN=OFF \
  -DDINGO_ENABLE_EIGEN_BLAS=ON \
  -DDINGO_ENABLE_EIGEN_LAPACK=ON \
  -DDINGO_BUILD_BENCHMARKS=ON

/Applications/CMake.app/Contents/bin/cmake --build build --target dingo-bench -- -j8
./build/benchmarks/dingo-bench
./build/benchmarks/dingo-bench --quick
./build/benchmarks/dingo-bench --quick --samples 7
./build/benchmarks/dingo-bench --sizes 128,256,512 --csv ./build/benchmarks/results.csv
```

Benchmark reports median time per operation (`*_ms/op`) and sample-to-sample jitter (`*_cv%`).

Covered benchmark classes:

- GEMM (`A * B`)
- Linear solve (`solve`)
- LU decomposition
- QR decomposition
- SVD decomposition
- Cholesky decomposition
- Symmetric eigendecomposition (`eig_sym`)

## License

- Dingo source code: MIT (`LICENSE`)
- Vendored Eigen and related third-party code: see `THIRD_PARTY_NOTICES.md` and `third_party/eigen/licenses/*`
- MPL-2.0 text is included at `LICENSES/MPL-2.0.txt`
