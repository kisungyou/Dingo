# Interop API

This page documents adapters from `interop/eigen.hpp`.

## `as_eigen(x)`

Overloads exist for:

- `Mat<T>`
- `Col<T>`
- `Row<T>`

(const and non-const)

- **What it does:** Returns a reference to the underlying Eigen object.
- **Inputs:** Dingo matrix/vector object.
- **Returns:** Eigen storage reference.
- **Simple example:**
  ```cpp
  dingo::mat A{{1,2},{3,4}};
  auto& E = dingo::as_eigen(A);
  E(0,0) = 42;
  ```
- **Common mistake:** Mutating through Eigen and forgetting it changes the Dingo object directly.

## `from_eigen_mat(x)`

- **What it does:** Converts Eigen matrix expression to `dingo::Mat`.
- **Inputs:** Any `Eigen::MatrixBase`-compatible matrix expression.
- **Returns:** `Mat<Scalar>`.
- **Simple example:**
  ```cpp
  Eigen::MatrixXd m(2,2);
  m << 1,2,3,4;
  auto dm = dingo::from_eigen_mat(m);
  ```
- **Common mistake:** Expecting lazy reference semantics; conversion produces Dingo-owned storage.

## `from_eigen_col(x)`

- **What it does:** Converts Eigen single-column matrix/vector to `dingo::Col`.
- **Inputs:** Eigen matrix expression with exactly one column.
- **Returns:** `Col<Scalar>`.
- **Simple example:**
  ```cpp
  Eigen::VectorXd v(3);
  v << 1,2,3;
  auto dv = dingo::from_eigen_col(v);
  ```
- **Common mistake:** Passing matrix with more than one column.

## `from_eigen_row(x)`

- **What it does:** Converts Eigen single-row matrix/vector to `dingo::Row`.
- **Inputs:** Eigen matrix expression with exactly one row.
- **Returns:** `Row<Scalar>`.
- **Simple example:**
  ```cpp
  Eigen::RowVectorXd r(3);
  r << 1,2,3;
  auto dr = dingo::from_eigen_row(r);
  ```
- **Common mistake:** Passing matrix with more than one row.
