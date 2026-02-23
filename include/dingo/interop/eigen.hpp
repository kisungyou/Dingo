#pragma once

#include "../core/matrix.hpp"
#include "../detail/assert.hpp"
#include "../detail/eigen_dense.hpp"

namespace dingo {

template <typename T>
inline typename Mat<T>::storage_type& as_eigen(Mat<T>& x) {
  return x.eigen();
}

template <typename T>
inline const typename Mat<T>::storage_type& as_eigen(const Mat<T>& x) {
  return x.eigen();
}

template <typename T>
inline typename Col<T>::storage_type& as_eigen(Col<T>& x) {
  return x.eigen();
}

template <typename T>
inline const typename Col<T>::storage_type& as_eigen(const Col<T>& x) {
  return x.eigen();
}

template <typename T>
inline typename Row<T>::storage_type& as_eigen(Row<T>& x) {
  return x.eigen();
}

template <typename T>
inline const typename Row<T>::storage_type& as_eigen(const Row<T>& x) {
  return x.eigen();
}

template <typename Derived>
inline Mat<typename Derived::Scalar> from_eigen_mat(const Eigen::MatrixBase<Derived>& x) {
  using scalar_t = typename Derived::Scalar;
  using storage_t = typename Mat<scalar_t>::storage_type;
  storage_t out = x.template cast<scalar_t>();
  return Mat<scalar_t>(std::move(out));
}

template <typename Derived>
inline Col<typename Derived::Scalar> from_eigen_col(const Eigen::MatrixBase<Derived>& x) {
  using scalar_t = typename Derived::Scalar;
  DINGO_ASSERT(x.cols() == 1, "dingo::from_eigen_col expects a single-column matrix");
  using storage_t = typename Col<scalar_t>::storage_type;
  storage_t out = x.template cast<scalar_t>();
  return Col<scalar_t>(std::move(out));
}

template <typename Derived>
inline Row<typename Derived::Scalar> from_eigen_row(const Eigen::MatrixBase<Derived>& x) {
  using scalar_t = typename Derived::Scalar;
  DINGO_ASSERT(x.rows() == 1, "dingo::from_eigen_row expects a single-row matrix");
  using storage_t = typename Row<scalar_t>::storage_type;
  storage_t out = x.template cast<scalar_t>();
  return Row<scalar_t>(std::move(out));
}

}  // namespace dingo
