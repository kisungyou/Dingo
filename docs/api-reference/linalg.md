# Linear Algebra API

This page covers functions in `linalg/decomp.hpp`.

## `solve(a, b)`

- **What it does:** Solves `a * x = b` for matrix or vector right-hand side.
- **Inputs:** Square `a`; `b` with matching row count.
- **Returns:** `Mat<T>` or `Col<T>` solution.
- **Simple example:**
  ```cpp
  dingo::mat A{{4,2},{1,3}};
  dingo::vec b{1,2};
  auto x = dingo::solve(A, b);
  ```
- **Common mistake:** Using non-square `a`.

## `inv(x)` and `pinv(x, tol=0)`

- **What it does:** Matrix inverse and pseudoinverse.
- **Inputs:** `inv` needs square matrix. `pinv` works on general shape, optional tolerance.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  auto Ainv = dingo::inv(A);
  auto Ap = dingo::pinv(A);
  ```
- **Common mistake:** Using `inv` on poorly conditioned matrix instead of `solve`/`pinv`.

## `det(x)`, `rank(x, tol=0)`, `cond(x)`

- **What it does:** Determinant, numerical rank, and condition number.
- **Inputs:** Matrix, optional tolerance for rank.
- **Returns:** Scalar values (`det` type `T`, `rank` type `uword`, `cond` real scalar).
- **Simple example:**
  ```cpp
  auto d = dingo::det(A);
  auto r = dingo::rank(A);
  auto c = dingo::cond(A);
  ```
- **Common mistake:** Interpreting very large `cond` as stable matrix.

## `eig_sym` (symmetric real eigen)

Overloads:
- `bool eig_sym(Col<T>& eigval, Mat<T>& eigvec, const Mat<T>& x)`
- `Col<T> eig_sym(const Mat<T>& x)`

- **What it does:** Eigendecomposition for symmetric real matrix.
- **Inputs:** Square symmetric real matrix.
- **Returns:** `bool` success + outputs, or eigenvalues only.
- **Simple example:**
  ```cpp
  dingo::vec eval;
  dingo::mat evec;
  bool ok = dingo::eig_sym(eval, evec, A);
  ```
- **Common mistake:** Passing non-symmetric matrix and trusting output.

## `eig_gen` (general real eigen)

Signature:
- `bool eig_gen(Col<std::complex<T>>& eigval, Mat<std::complex<T>>& eigvec, const Mat<T>& x)`

- **What it does:** General eigen decomposition for real square matrix.
- **Inputs:** Square real matrix.
- **Returns:** `bool` and complex eigenvalues/eigenvectors.
- **Simple example:**
  ```cpp
  dingo::cx_vec eval;
  dingo::cx_mat evec;
  bool ok = dingo::eig_gen(eval, evec, A);
  ```
- **Common mistake:** Expecting purely real output for non-symmetric input.

## `svd`

Overloads:
- `bool svd(Mat<T>& u, Col<T>& s, Mat<T>& v, const Mat<T>& x)`
- `Col<T> svd(const Mat<T>& x)`

- **What it does:** Singular value decomposition.
- **Inputs:** Any matrix.
- **Returns:** `bool` + factors, or singular values only.
- **Simple example:**
  ```cpp
  dingo::mat U, V;
  dingo::vec S;
  dingo::svd(U, S, V, A);
  ```
- **Common mistake:** Assuming singular values are returned as matrix instead of column vector.

## `qr(q, r, x)`

- **What it does:** QR decomposition.
- **Inputs:** Matrix `x`.
- **Returns:** `bool` success, outputs `q` and `r`.
- **Simple example:**
  ```cpp
  dingo::mat Q, R;
  dingo::qr(Q, R, A);
  ```
- **Common mistake:** Forgetting to check returned `bool`.

## `chol(out, x, layout="upper")`

- **What it does:** Cholesky decomposition for SPD matrix.
- **Inputs:** Square matrix `x`, optional layout (`"upper"` or `"lower"`).
- **Returns:** `bool` success and triangular factor.
- **Simple example:**
  ```cpp
  dingo::mat U;
  if (dingo::chol(U, A)) {
    // A = trans(U) * U
  }
  ```
- **Common mistake:** Running `chol` on non-SPD matrix and ignoring failure.

## `lu(l, u, p, x)`

- **What it does:** LU decomposition with permutation matrix.
- **Inputs:** Square matrix `x` (v1).
- **Returns:** `bool` plus `L`, `U`, `P`.
- **Simple example:**
  ```cpp
  dingo::mat L, U, P;
  dingo::lu(L, U, P, A);
  ```
- **Common mistake:** Assuming rectangular support in v1.

## `log_det(log_value, sign, x)`

- **What it does:** Computes log absolute determinant and sign robustly.
- **Inputs:** Square matrix `x`.
- **Returns:** `bool` invertibility; updates `log_value`, `sign`.
- **Simple example:**
  ```cpp
  double lv = 0.0, s = 0.0;
  bool ok = dingo::log_det(lv, s, A);
  ```
- **Common mistake:** Ignoring `ok` when matrix is singular.

## `kron(a, b)`

- **What it does:** Kronecker product.
- **Inputs:** Two matrices.
- **Returns:** Expanded matrix.
- **Simple example:**
  ```cpp
  auto K = dingo::kron(A, B);
  ```
- **Common mistake:** Underestimating output size growth.

## `null(x, tol=0)`

- **What it does:** Basis for null space of `x`.
- **Inputs:** Matrix and optional tolerance.
- **Returns:** Matrix whose columns span null space.
- **Simple example:**
  ```cpp
  auto N = dingo::null(A);
  ```
- **Common mistake:** Expecting non-empty output for full-column-rank matrix.

## `orth(x, tol=0)` and `ortho(x, tol=0)`

- **What it does:** Orthonormal basis for column space (`ortho` is alias).
- **Inputs:** Matrix and optional tolerance.
- **Returns:** Basis matrix.
- **Simple example:**
  ```cpp
  auto Q = dingo::orth(A);
  auto Q2 = dingo::ortho(A);
  ```
- **Common mistake:** Assuming returned basis has same column count as input.
