#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include "../core/functions.hpp"
#include "../detail/assert.hpp"
#include "../detail/eigen_dense.hpp"

namespace dingo {

template <typename T>
inline Mat<T> solve(const Mat<T>& a, const Mat<T>& b) {
  DINGO_ASSERT(a.n_rows() == a.n_cols(), "dingo::solve requires square lhs");
  DINGO_ASSERT(a.n_rows() == b.n_rows(), "dingo::solve dimension mismatch");
  typename Mat<T>::storage_type out = a.eigen().colPivHouseholderQr().solve(b.eigen());
  return Mat<T>(std::move(out));
}

template <typename T>
inline Col<T> solve(const Mat<T>& a, const Col<T>& b) {
  DINGO_ASSERT(a.n_rows() == a.n_cols(), "dingo::solve requires square lhs");
  DINGO_ASSERT(a.n_rows() == b.n_rows(), "dingo::solve dimension mismatch");
  typename Col<T>::storage_type out = a.eigen().colPivHouseholderQr().solve(b.eigen());
  return Col<T>(std::move(out));
}

template <typename T>
inline Mat<T> inv(const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::inv requires square matrix");
  return Mat<T>(typename Mat<T>::storage_type(x.eigen().inverse()));
}

template <typename T>
inline Mat<T> pinv(const Mat<T>& x, typename Eigen::NumTraits<T>::Real tol = 0) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  Eigen::JacobiSVD<typename Mat<T>::storage_type> svd(
      x.eigen(), Eigen::ComputeThinU | Eigen::ComputeThinV);

  const auto singular = svd.singularValues();
  const real_t default_tol = std::numeric_limits<real_t>::epsilon() *
                             static_cast<real_t>(std::max(x.n_rows(), x.n_cols())) *
                             (singular.size() > 0 ? singular(0) : real_t(0));
  const real_t effective_tol = (tol > real_t(0)) ? tol : default_tol;

  typename Mat<T>::storage_type s_pinv =
      Mat<T>::storage_type::Zero(svd.matrixV().cols(), svd.matrixU().cols());
  for (Eigen::Index i = 0; i < singular.size(); ++i) {
    if (singular(i) > effective_tol) {
      s_pinv(i, i) = T(1) / singular(i);
    }
  }

  typename Mat<T>::storage_type out = svd.matrixV() * s_pinv * svd.matrixU().adjoint();
  return Mat<T>(std::move(out));
}

template <typename T>
inline T det(const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::det requires square matrix");
  return x.eigen().determinant();
}

template <typename T>
inline uword rank(const Mat<T>& x, typename Eigen::NumTraits<T>::Real tol = 0) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  Eigen::JacobiSVD<typename Mat<T>::storage_type> svd(x.eigen());
  const auto singular = svd.singularValues();
  const real_t default_tol = std::numeric_limits<real_t>::epsilon() *
                             static_cast<real_t>(std::max(x.n_rows(), x.n_cols())) *
                             (singular.size() > 0 ? singular(0) : real_t(0));
  const real_t effective_tol = (tol > real_t(0)) ? tol : default_tol;
  return static_cast<uword>((singular.array() > effective_tol).count());
}

template <typename T>
inline typename Eigen::NumTraits<T>::Real cond(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  Eigen::JacobiSVD<typename Mat<T>::storage_type> svd(x.eigen());
  const auto singular = svd.singularValues();
  if (singular.size() == 0) {
    return real_t(0);
  }
  const real_t max_sv = singular(0);
  const real_t min_sv = singular(singular.size() - 1);
  if (min_sv == real_t(0)) {
    return std::numeric_limits<real_t>::infinity();
  }
  return max_sv / min_sv;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool eig_sym(Col<T>& eigval, Mat<T>& eigvec, const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::eig_sym requires square matrix");
  Eigen::SelfAdjointEigenSolver<typename Mat<T>::storage_type> solver(x.eigen());
  if (solver.info() != Eigen::Success) {
    return false;
  }
  eigvec = Mat<T>(typename Mat<T>::storage_type(solver.eigenvectors()));
  eigval = Col<T>(typename Col<T>::storage_type(solver.eigenvalues()));
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline Col<T> eig_sym(const Mat<T>& x) {
  Col<T> eigval;
  Mat<T> eigvec;
  if (!eig_sym(eigval, eigvec, x)) {
    return Col<T>();
  }
  return eigval;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool eig_gen(Col<std::complex<T>>& eigval, Mat<std::complex<T>>& eigvec, const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::eig_gen requires square matrix");
  Eigen::EigenSolver<typename Mat<T>::storage_type> solver(x.eigen());
  if (solver.info() != Eigen::Success) {
    return false;
  }

  typename Col<std::complex<T>>::storage_type vals = solver.eigenvalues();
  typename Mat<std::complex<T>>::storage_type vecs = solver.eigenvectors();
  eigval = Col<std::complex<T>>(std::move(vals));
  eigvec = Mat<std::complex<T>>(std::move(vecs));
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool svd(Mat<T>& u, Col<T>& s, Mat<T>& v, const Mat<T>& x) {
  Eigen::JacobiSVD<typename Mat<T>::storage_type> solver(
      x.eigen(), Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  u = Mat<T>(typename Mat<T>::storage_type(solver.matrixU()));
  v = Mat<T>(typename Mat<T>::storage_type(solver.matrixV()));
  s = Col<T>(typename Col<T>::storage_type(solver.singularValues()));
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline Col<T> svd(const Mat<T>& x) {
  Mat<T> u;
  Mat<T> v;
  Col<T> s;
  if (!svd(u, s, v, x)) {
    return Col<T>();
  }
  return s;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool qr(Mat<T>& q, Mat<T>& r, const Mat<T>& x) {
  Eigen::HouseholderQR<typename Mat<T>::storage_type> solver(x.eigen());
  if (solver.info() != Eigen::Success) {
    return false;
  }

  const auto m = static_cast<Eigen::Index>(x.n_rows());
  typename Mat<T>::storage_type ident = Mat<T>::storage_type::Identity(m, m);
  typename Mat<T>::storage_type q_out = solver.householderQ() * ident;
  typename Mat<T>::storage_type r_out = solver.matrixQR().template triangularView<Eigen::Upper>();
  q = Mat<T>(std::move(q_out));
  r = Mat<T>(std::move(r_out));
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool chol(Mat<T>& out, const Mat<T>& x, const char* layout = "upper") {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::chol requires square matrix");
  Eigen::LLT<typename Mat<T>::storage_type> solver(x.eigen());
  if (solver.info() != Eigen::Success) {
    return false;
  }
  if (layout[0] == 'l' || layout[0] == 'L') {
    out = Mat<T>(typename Mat<T>::storage_type(solver.matrixL()));
  } else {
    out = Mat<T>(typename Mat<T>::storage_type(solver.matrixU()));
  }
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool lu(Mat<T>& l, Mat<T>& u, Mat<T>& p, const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::lu requires square matrix in v1");
  Eigen::PartialPivLU<typename Mat<T>::storage_type> solver(x.eigen());

  const auto n = static_cast<Eigen::Index>(x.n_rows());
  typename Mat<T>::storage_type lu_mat = solver.matrixLU();
  typename Mat<T>::storage_type l_out = Mat<T>::storage_type::Identity(n, n);
  l_out.template triangularView<Eigen::StrictlyLower>() = lu_mat.template triangularView<Eigen::StrictlyLower>();
  typename Mat<T>::storage_type u_out = lu_mat.template triangularView<Eigen::Upper>();
  typename Mat<T>::storage_type p_out = solver.permutationP().toDenseMatrix().template cast<T>();

  l = Mat<T>(std::move(l_out));
  u = Mat<T>(std::move(u_out));
  p = Mat<T>(std::move(p_out));
  return true;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline bool log_det(T& log_value, T& sign, const Mat<T>& x) {
  DINGO_ASSERT(x.n_rows() == x.n_cols(), "dingo::log_det requires square matrix");
  Eigen::FullPivLU<typename Mat<T>::storage_type> solver(x.eigen());
  if (!solver.isInvertible()) {
    log_value = -std::numeric_limits<T>::infinity();
    sign = T(0);
    return false;
  }
  const T d = solver.determinant();
  sign = (d < T(0)) ? T(-1) : T(1);
  log_value = std::log(std::abs(d));
  return true;
}

template <typename T>
inline Mat<T> kron(const Mat<T>& a, const Mat<T>& b) {
  const auto out_rows = a.n_rows() * b.n_rows();
  const auto out_cols = a.n_cols() * b.n_cols();
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(out_rows), static_cast<Eigen::Index>(out_cols));
  for (uword j = 0; j < a.n_cols(); ++j) {
    for (uword i = 0; i < a.n_rows(); ++i) {
      out.block(static_cast<Eigen::Index>(i * b.n_rows()),
                static_cast<Eigen::Index>(j * b.n_cols()),
                static_cast<Eigen::Index>(b.n_rows()),
                static_cast<Eigen::Index>(b.n_cols())) = a(i, j) * b.eigen();
    }
  }
  return Mat<T>(std::move(out));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline Mat<T> null(const Mat<T>& x, typename Eigen::NumTraits<T>::Real tol = 0) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  Eigen::JacobiSVD<typename Mat<T>::storage_type> svd(x.eigen(), Eigen::ComputeFullV);
  const auto singular = svd.singularValues();
  const real_t default_tol = std::numeric_limits<real_t>::epsilon() *
                             static_cast<real_t>(std::max(x.n_rows(), x.n_cols())) *
                             (singular.size() > 0 ? singular(0) : real_t(0));
  const real_t effective_tol = (tol > real_t(0)) ? tol : default_tol;
  const Eigen::Index r = (singular.array() > effective_tol).count();
  const Eigen::Index n = static_cast<Eigen::Index>(x.n_cols());
  const Eigen::Index k = n - r;
  if (k <= 0) {
    return Mat<T>(x.n_cols(), 0);
  }
  typename Mat<T>::storage_type out = svd.matrixV().rightCols(k);
  return Mat<T>(std::move(out));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline Mat<T> orth(const Mat<T>& x, typename Eigen::NumTraits<T>::Real tol = 0) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  Eigen::JacobiSVD<typename Mat<T>::storage_type> svd(x.eigen(), Eigen::ComputeThinU);
  const auto singular = svd.singularValues();
  const real_t default_tol = std::numeric_limits<real_t>::epsilon() *
                             static_cast<real_t>(std::max(x.n_rows(), x.n_cols())) *
                             (singular.size() > 0 ? singular(0) : real_t(0));
  const real_t effective_tol = (tol > real_t(0)) ? tol : default_tol;
  const Eigen::Index r = (singular.array() > effective_tol).count();
  if (r <= 0) {
    return Mat<T>(x.n_rows(), 0);
  }
  typename Mat<T>::storage_type out = svd.matrixU().leftCols(r);
  return Mat<T>(std::move(out));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline Mat<T> ortho(const Mat<T>& x, typename Eigen::NumTraits<T>::Real tol = 0) {
  return orth(x, tol);
}

}  // namespace dingo
