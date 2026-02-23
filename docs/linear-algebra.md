# Linear Algebra

This page explains when to use each decomposition/solver in Dingo.

## Solve `Ax = b`

Use `solve` for standard linear systems.

```cpp
dingo::mat A{{4.0, 2.0}, {1.0, 3.0}};
dingo::vec b{1.0, 2.0};
dingo::vec x = dingo::solve(A, b);
```

## Inverse and Pseudoinverse

- `inv(A)`: exact inverse of square, invertible matrix.
- `pinv(A)`: Moore-Penrose pseudoinverse for rank-deficient/rectangular cases.

## Matrix Diagnostics

- `det(A)`: determinant
- `rank(A)`: numerical rank
- `cond(A)`: condition number
- `log_det(logv, sign, A)`: stable log-determinant output

## Eigendecomposition

- `eig_sym(...)`: for symmetric real matrices.
- `eig_gen(...)`: for general real matrices (complex eigenvalues/eigenvectors).

## SVD (Singular Value Decomposition)

Use `svd` for robust analysis, rank, compression, and least-squares style workflows.

## QR

Use `qr` for orthogonal-triangular decomposition, often used in least squares.

## Cholesky

Use `chol` for symmetric positive definite (SPD) matrices.

- `chol(U, A)` gives upper triangular by default.
- `chol(L, A, "lower")` gives lower triangular.

## LU

Use `lu(L, U, P, A)` for square matrix factorization with pivoting.

## Kronecker, Null Space, Orthonormal Basis

- `kron(A, B)`: Kronecker product
- `null(A)`: basis for null space
- `orth(A)`: orthonormal basis for column space
