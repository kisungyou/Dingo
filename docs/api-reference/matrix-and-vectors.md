# Matrix and Vectors API

## Types and Aliases

### `Mat<T>`, `Col<T>`, `Row<T>`

- **What it does:** Core dense matrix/vector containers.
- **Inputs:** Template type `T` (usually `double` or `std::complex<double>`).
- **Returns:** Container objects with Armadillo-style API.
- **Simple example:**
  ```cpp
  dingo::Mat<double> A(2, 2);
  dingo::Col<double> v(2);
  dingo::Row<double> r(2);
  ```
- **Common mistake:** Mixing incompatible shapes for arithmetic.

### Common aliases

- **What it does:** Short names for common scalar types.
- **Inputs:** None.
- **Returns:** Type aliases.
- **Simple example:**
  ```cpp
  dingo::mat A;         // Mat<double>
  dingo::vec v;         // Col<double>
  dingo::rowvec r;      // Row<double>
  dingo::cx_mat C;      // Mat<std::complex<double>>
  ```
- **Common mistake:** Assuming `vec` is row-oriented; it is a column vector.

## Matrix Construction and Access (`Mat<T>`)

### Constructors

- **What it does:** Create matrices by size, fill mode, or initializer list.
- **Inputs:** `(rows, cols)`, fill tag (`fill::zeros`, `fill::ones`, `fill::randu`, `fill::randn`), or `{{...}, {...}}`.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  dingo::mat A(3, 4, dingo::fill::zeros);
  dingo::mat B{{1,2,3},{4,5,6}};
  ```
- **Common mistake:** Assuming jagged initializer rows preserve jagged shape; missing values are zero-filled.

### Size and state

APIs: `n_rows()`, `n_cols()`, `n_elem()`, `is_empty()`, `resize(rows, cols)`.

- **What it does:** Query or change shape.
- **Inputs:** Optional new dimensions for `resize`.
- **Returns:** Counts or `void`.
- **Simple example:**
  ```cpp
  if (!A.is_empty()) {
    auto n = A.n_elem();
  }
  A.resize(10, 2);
  ```
- **Common mistake:** Expecting `resize` to preserve old values in a specific layout.

### Fill mutators

APIs: `zeros()`, `ones()`, `randu()`, `randn()`.

- **What it does:** Refill matrix values in-place.
- **Inputs:** None.
- **Returns:** `void`.
- **Simple example:**
  ```cpp
  A.zeros();
  A.randn();
  ```
- **Common mistake:** Using uninitialized matrix dimensions before filling.

### Element access

APIs: `operator()(row, col)`, `operator()(index)`, `at(row, col)`.

- **What it does:** Read/write by 2D or linear index.
- **Inputs:** Indices.
- **Returns:** Element reference/value.
- **Simple example:**
  ```cpp
  A(0,1) = 3.0;
  double x = A(0,1);
  double y = A(2);  // linear index (column-major)
  ```
- **Common mistake:** Assuming row-major linear indexing.

### Row/column/submatrix helpers

APIs: `row(i)`, `col(j)`, `submat(r1,c1,r2,c2)`, `submat(span,span)`, `set_row`, `set_col`, `set_submat`.

- **What it does:** Extract or assign matrix regions.
- **Inputs:** Indices, ranges, block matrices.
- **Returns:** New `Mat<T>` or mutates in-place.
- **Simple example:**
  ```cpp
  auto r1 = A.row(1);
  auto block = A.submat(dingo::range(0,1), dingo::range(1,2));
  A.set_col(0, dingo::mat{{1},{2},{3}});
  ```
- **Common mistake:** Block assignment with mismatched shape.

### Span and all-index views

APIs: `range(a,b)`, `all_idx` / `all_t`, `operator()(span, span)`, `operator()(all_t, span)`, `operator()(span, all_t)`.

- **What it does:** MATLAB-style slicing and assignment.
- **Inputs:** Range objects or all-index markers.
- **Returns:** Submatrix values or writable subview proxy.
- **Simple example:**
  ```cpp
  A(dingo::range(1,2), dingo::range(0,1)) = dingo::mat{{9,9},{9,9}};
  ```
- **Common mistake:** Using out-of-range spans.

### Mask indexing

APIs: `operator()(mask)` (const and mutable).

- **What it does:** Select/assign elements where mask is non-zero.
- **Inputs:** Mask matrix with same shape.
- **Returns:** Selected values as `Nx1` matrix or writable proxy.
- **Simple example:**
  ```cpp
  dingo::mat mask{{0,1},{1,0}};
  auto vals = A(mask);
  A(mask) = -1.0;
  ```
- **Common mistake:** Mask shape mismatch.

### Transpose helpers

APIs: `t()`, `st()`.

- **What it does:** Regular transpose (`t`) and conjugate transpose (`st`).
- **Inputs:** None.
- **Returns:** New `Mat<T>`.
- **Simple example:**
  ```cpp
  auto At = A.t();
  auto Ah = A.st();
  ```
- **Common mistake:** Using `t()` when complex conjugation is required.

### Eigen storage access

APIs: `eigen()` const/non-const.

- **What it does:** Returns underlying Eigen storage object.
- **Inputs:** None.
- **Returns:** Eigen matrix reference.
- **Simple example:**
  ```cpp
  A.eigen()(0,0) = 1.0;
  ```
- **Common mistake:** Mutating through Eigen then assuming no side effects in Dingo object.

## Vector APIs (`Col<T>`, `Row<T>`)

### Construction and fill

APIs: constructors by size/fill/initializer list, `zeros()`, `ones()`, `randu()`, `randn()`.

- **What it does:** Build and fill vectors.
- **Inputs:** Element count, fill tags, initializer values.
- **Returns:** Vector objects.
- **Simple example:**
  ```cpp
  dingo::vec v{1.0,2.0,3.0};
  dingo::rowvec r(3, dingo::fill::ones);
  ```
- **Common mistake:** Assigning rowvector where colvector is expected.

### Access and conversion

APIs: `operator()(i)`, `n_elem()`, `as_mat()`, `eigen()`.

- **What it does:** Access vector entries and convert to matrix view/copy.
- **Inputs:** Index (for access).
- **Returns:** Element values, size counts, or `Mat<T>`.
- **Simple example:**
  ```cpp
  double x = v(0);
  auto vm = v.as_mat(); // Nx1 matrix
  ```
- **Common mistake:** Expecting `as_mat()` to become row matrix for `Col`.

## Arithmetic Operators

### Matrix-matrix

APIs: `+`, `-`, `%`, `*`, `/`, unary `-`, compound assignments (`+=`, `-=`, `%=`, `*=`, `/=` scalar).

- **What it does:** Standard algebra, Hadamard product, matrix multiply, elementwise divide.
- **Inputs:** Compatible shapes or scalars.
- **Returns:** New matrix or in-place mutation.
- **Simple example:**
  ```cpp
  auto C = A * B;
  auto H = A % B;
  auto D = A / B; // elementwise
  ```
- **Common mistake:** Confusing `%` (elementwise) with `*` (matrix multiply).

### Broadcasting with vectors

APIs: `Mat +/-/%// Col`, `Mat +/-/%// Row`.

- **What it does:** Broadcast row-wise or column-wise vector operations.
- **Inputs:** Matrix and matching `Col` or `Row` length.
- **Returns:** New matrix.
- **Simple example:**
  ```cpp
  auto out1 = A + dingo::vec{1,2,3};
  auto out2 = A + dingo::rowvec{10,20};
  ```
- **Common mistake:** Wrong vector length relative to matrix dimension.
