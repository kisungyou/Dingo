# Functions API

This page covers free functions from `core/functions.hpp`.

## Creation Helpers

### `zeros`, `ones`, `randu`, `randn`, `eye`

- **What it does:** Creates matrices/vectors with standard initial values.
- **Inputs:** Matrix shape (`rows`, `cols`) or vector length (`n_elem`).
- **Returns:** `Mat<T>` or `Col<T>`.
- **Simple example:**
  ```cpp
  auto Z = dingo::zeros(2, 3);
  auto v = dingo::ones(5);
  auto I = dingo::eye(4);
  ```
- **Common mistake:** Passing vector length to matrix overload accidentally.

### `linspace(start, end, n)` / `regspace(start, step, end)`

- **What it does:** Create numeric sequences as column vectors.
- **Inputs:** Start/end/step/count values.
- **Returns:** `Col<T>`.
- **Simple example:**
  ```cpp
  auto a = dingo::linspace(0.0, 1.0, 5);
  auto b = dingo::regspace(1.0, 2.0, 7.0);
  ```
- **Common mistake:** Using zero step in `regspace`.

## Shape and Arrangement

### `trans`, `strans`, `reshape`, `vectorise`

- **What it does:** Transpose and reshape matrix data.
- **Inputs:** Matrix and target shape where needed.
- **Returns:** Transformed matrix/vector.
- **Simple example:**
  ```cpp
  auto t = dingo::trans(A);
  auto v = dingo::vectorise(A);
  auto r = dingo::reshape(v.as_mat(), 3, 2);
  ```
- **Common mistake:** `reshape` with mismatched total element count.

### `join_rows`, `join_cols`

- **What it does:** Concatenate matrices horizontally or vertically.
- **Inputs:** Two matrices with compatible dimensions.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  auto H = dingo::join_rows(A, B);
  auto V = dingo::join_cols(A, B);
  ```
- **Common mistake:** Row mismatch for `join_rows` or column mismatch for `join_cols`.

## Reductions and Statistics

### `sum`, `accu`, `mean`

- **What it does:** Sum and mean over all values or by dimension (`dim=0/1`).
- **Inputs:** Matrix/vector and optional `dim`.
- **Returns:** Scalar or matrix of reduced values.
- **Simple example:**
  ```cpp
  double s = dingo::sum(A);
  auto cs = dingo::sum(A, 0);
  auto rs = dingo::sum(A, 1);
  ```
- **Common mistake:** Using invalid dimension value.

### `min`, `max`, `var`, `stddev`

- **What it does:** Scalar and dimension-wise statistics.
- **Inputs:** Matrix and optional `dim`.
- **Returns:** Scalar or reduction matrix.
- **Simple example:**
  ```cpp
  double lo = dingo::min(A);
  auto col_var = dingo::var(A, 0);
  ```
- **Common mistake:** Calling dim-wise variance on too-small dimension (e.g., one row).

### `norm`, `trace`

- **What it does:** Matrix norm (`p=0,1,2`) and trace.
- **Inputs:** Matrix and optional `p`.
- **Returns:** Scalar.
- **Simple example:**
  ```cpp
  double n2 = dingo::norm(A, 2);
  double tr = dingo::trace(A);
  ```
- **Common mistake:** Using unsupported norm order (only `0`, `1`, `2` in v1).

## Structural Helpers

### `diagvec`, `diagmat`, `tril`, `triu`

- **What it does:** Diagonal extraction/build and triangular filtering.
- **Inputs:** Matrix/vector and optional diagonal offset `k` for `tril/triu`.
- **Returns:** `Col<T>` or `Mat<T>`.
- **Simple example:**
  ```cpp
  auto d = dingo::diagvec(A);
  auto D = dingo::diagmat(d);
  auto L = dingo::tril(A);
  auto U = dingo::triu(A, 1);
  ```
- **Common mistake:** Expecting `diagmat(A)` to preserve off-diagonal entries.

### `flipud`, `fliplr`, `repmat`

- **What it does:** Reverse matrix orientation and tile matrices.
- **Inputs:** Matrix and repetition counts for `repmat`.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  auto updown = dingo::flipud(A);
  auto tiled = dingo::repmat(A, 2, 3);
  ```
- **Common mistake:** Large replication counts causing unexpectedly large memory usage.

## Elementwise Math

### `sin`, `cos`, `exp`, `log`, `sqrt`

- **What it does:** Applies scalar math function to every element.
- **Inputs:** Matrix.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  auto E = dingo::exp(A);
  auto R = dingo::sqrt(dingo::mat{{4.0}});
  ```
- **Common mistake:** Domain issues (`log` of non-positive values, `sqrt` of negative real values).

### Complex helpers: `abs`, `conj`, `real`, `imag`, `angle`, `abs2`

- **What it does:** Complex-valued elementwise transforms.
- **Inputs:** Real/complex matrix.
- **Returns:** Real or complex matrix depending on function.
- **Simple example:**
  ```cpp
  dingo::cx_mat C{{{3.0, 4.0}}};
  auto mag = dingo::abs(C);
  auto ph = dingo::angle(C);
  ```
- **Common mistake:** Expecting `abs` to return complex values; it returns real magnitudes.

## Logical and Index Helpers

### `any`, `all`

- **What it does:** Dimension-wise nonzero checks.
- **Inputs:** Matrix and `dim` (`0` or `1`).
- **Returns:** Numeric mask matrix (`1` or `0` values).
- **Simple example:**
  ```cpp
  auto c_any = dingo::any(A, 0);
  auto r_all = dingo::all(A, 1);
  ```
- **Common mistake:** Treating output as boolean C++ type instead of numeric matrix.

### `sort`, `unique`

- **What it does:** Sort vectors and remove duplicates (column-vector version for `unique`).
- **Inputs:** `Col<T>` or `Row<T>`.
- **Returns:** Sorted vector or unique `Col<T>`.
- **Simple example:**
  ```cpp
  dingo::vec v{3,1,2,1};
  auto s = dingo::sort(v);
  auto u = dingo::unique(v);
  ```
- **Common mistake:** Expecting `unique` to preserve original order.

### `find`, `elem`

- **What it does:** Find non-zero linear indices and gather elements by index.
- **Inputs:** Matrix and optional index vector (`Col<uword>`).
- **Returns:** Index vector or selected value vector.
- **Simple example:**
  ```cpp
  auto idx = dingo::find(A);
  auto vals = dingo::elem(A, idx);
  ```
- **Common mistake:** Passing out-of-range indices to `elem`.

## Cube Utilities

### `squeeze`, `permute`, `join_slices`

- **What it does:** Convert/reorder/combine cube data.
- **Inputs:** `Cube<T>`, permutation dims (1-based), or two cubes.
- **Returns:** `Mat<T>` or `Cube<T>`.
- **Simple example:**
  ```cpp
  dingo::cube C(2,3,1, dingo::fill::zeros);
  auto M = dingo::squeeze(C);
  auto P = dingo::permute(C, 1, 3, 2);
  ```
- **Common mistake:** Using repeated or out-of-range dims in `permute`.
