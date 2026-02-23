# Dingo Documentation

Dingo is a header-only C++ numerical library with an Armadillo-style API and an Eigen backend.

This site is written for people who want to use Dingo quickly, including users with limited C++ background.

## Who This Is For

- You want matrix/vector operations in C++ with a simple API.
- You prefer practical examples over implementation details.
- You want Armadillo-like naming and workflow.

## First 10 Minutes

1. Build Dingo and run tests: see [Getting Started](getting-started.md).
2. Create a matrix and vector.
3. Solve one linear system (`Ax = b`).
4. Try one indexing example.
5. Jump to [API Reference](api-reference/index.md) for more functions.

## I Want To Do X

| Task | Go to |
|---|---|
| Create a matrix | [`zeros`, `ones`, constructors](basic-operations.md#construction-and-fill) |
| Solve `Ax = b` | [`solve`](linear-algebra.md#solve-ax--b) |
| Compute SVD | [`svd`](linear-algebra.md#svd-singular-value-decomposition) |
| Transpose a matrix | [`trans`, `strans`](basic-operations.md#transpose) |
| Use Eigen + Dingo together | [Eigen Interop](eigen-interop.md) |
| See all API entries | [API Reference](api-reference/index.md) |

## Current Scope

Dingo currently does **not** include:

- File I/O APIs
- Sparse matrix APIs
- `kmeans` and Gaussian mixture model estimation
