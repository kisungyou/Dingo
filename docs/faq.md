# FAQ

## Is Dingo a replacement for Eigen?

No. Dingo is an easier front-end API, and Eigen is still the backend.

## Why does `A(i)` not match row-major order?

Dingo uses column-major linear indexing, similar to Armadillo and MATLAB conventions.

## What is the difference between `%` and `*`?

- `%` is elementwise multiplication.
- `*` is matrix multiplication.

## When should I use `solve` instead of `inv`?

Use `solve` for linear systems (`Ax=b`). It is usually more numerically stable than computing `inv(A)` first.

## Why did `chol` return false?

`chol` requires a symmetric positive definite (SPD) matrix.

## Why does `squeeze` fail on my cube?

`squeeze` currently supports only cubes with exactly one slice.

## Glossary

- **elementwise:** Apply operation independently to each element.
- **broadcasting:** Apply vector operation across rows or columns of a matrix.
- **SPD:** Symmetric positive definite matrix.
- **rank-deficient:** Matrix has dependent rows/columns and less than full rank.
- **slice:** One 2D matrix inside a `Cube`.
- **submatrix:** A selected rectangular region from a matrix.
