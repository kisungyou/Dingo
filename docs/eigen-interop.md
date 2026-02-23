# Eigen Interop

Dingo is backed by Eigen, and provides explicit adapters so you can combine both libraries.

## Access Eigen Storage

Use `as_eigen` to access the underlying Eigen object.

```cpp
dingo::mat A{{1.0, 2.0}, {3.0, 4.0}};
auto& E = dingo::as_eigen(A);
E(0, 0) = 42.0;
```

## Convert From Eigen

Use conversion helpers when you already have Eigen values.

```cpp
Eigen::MatrixXd m(2,2);
m << 1.0, 2.0, 3.0, 4.0;
auto dm = dingo::from_eigen_mat(m);

Eigen::VectorXd v(3);
v << 5.0, 6.0, 7.0;
auto dc = dingo::from_eigen_col(v);

Eigen::RowVectorXd r(3);
r << 8.0, 9.0, 10.0;
auto dr = dingo::from_eigen_row(r);
```

## Caveats

- `from_eigen_col` requires a single-column matrix.
- `from_eigen_row` requires a single-row matrix.
- Shape mismatch triggers assertions.
