#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

#include "matrix.hpp"
#include "../detail/assert.hpp"

namespace dingo {

template <typename T = double>
inline Mat<T> zeros(uword rows, uword cols) {
  return Mat<T>(rows, cols, fill::zeros);
}

template <typename T = double>
inline Col<T> zeros(uword n_elem) {
  return Col<T>(n_elem, fill::zeros);
}

template <typename T = double>
inline Mat<T> ones(uword rows, uword cols) {
  return Mat<T>(rows, cols, fill::ones);
}

template <typename T = double>
inline Col<T> ones(uword n_elem) {
  return Col<T>(n_elem, fill::ones);
}

template <typename T = double>
inline Mat<T> randu(uword rows, uword cols) {
  return Mat<T>(rows, cols, fill::randu);
}

template <typename T = double>
inline Col<T> randu(uword n_elem) {
  return Col<T>(n_elem, fill::randu);
}

template <typename T = double>
inline Mat<T> randn(uword rows, uword cols) {
  return Mat<T>(rows, cols, fill::randn);
}

template <typename T = double>
inline Col<T> randn(uword n_elem) {
  return Col<T>(n_elem, fill::randn);
}

template <typename T = double>
inline Mat<T> eye(uword rows, uword cols) {
  typename Mat<T>::storage_type out =
      Mat<T>::storage_type::Identity(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  return Mat<T>(std::move(out));
}

template <typename T = double>
inline Mat<T> eye(uword n) {
  return eye<T>(n, n);
}

template <typename T>
inline Col<T> linspace(T start, T end, uword n) {
  Col<T> out(n);
  if (n == 0) {
    return out;
  }
  if (n == 1) {
    out(0) = start;
    return out;
  }
  const auto step = (end - start) / static_cast<T>(n - 1);
  for (uword i = 0; i < n; ++i) {
    out(i) = start + static_cast<T>(i) * step;
  }
  return out;
}

template <typename T>
inline Col<T> regspace(T start, T step, T end) {
  DINGO_ASSERT(step != T(0), "dingo::regspace step cannot be zero");
  uword n = 0;
  if ((step > T(0) && start <= end) || (step < T(0) && start >= end)) {
    const auto span = (end - start) / step;
    n = static_cast<uword>(std::floor(static_cast<double>(span))) + 1;
  }

  Col<T> out(n);
  for (uword i = 0; i < n; ++i) {
    out(i) = start + static_cast<T>(i) * step;
  }
  return out;
}

template <typename T>
inline Mat<T> trans(const Mat<T>& x) {
  return x.t();
}

template <typename T>
inline Mat<T> strans(const Mat<T>& x) {
  return x.st();
}

template <typename T>
inline Mat<T> reshape(const Mat<T>& x, uword rows, uword cols) {
  DINGO_ASSERT(x.n_elem() == rows * cols, "dingo::reshape total elements mismatch");
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  out = Eigen::Map<const typename Mat<T>::storage_type>(
      x.eigen().data(), static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  return Mat<T>(std::move(out));
}

template <typename T>
inline Col<T> vectorise(const Mat<T>& x) {
  typename Col<T>::storage_type out(static_cast<Eigen::Index>(x.n_elem()));
  out = Eigen::Map<const typename Col<T>::storage_type>(x.eigen().data(), static_cast<Eigen::Index>(x.n_elem()));
  return Col<T>(std::move(out));
}

template <typename T>
inline Mat<T> join_rows(const Mat<T>& a, const Mat<T>& b) {
  DINGO_ASSERT(a.n_rows() == b.n_rows(), "dingo::join_rows row mismatch");
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(a.n_rows()),
                                    static_cast<Eigen::Index>(a.n_cols() + b.n_cols()));
  out << a.eigen(), b.eigen();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> join_cols(const Mat<T>& a, const Mat<T>& b) {
  DINGO_ASSERT(a.n_cols() == b.n_cols(), "dingo::join_cols col mismatch");
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(a.n_rows() + b.n_rows()),
                                    static_cast<Eigen::Index>(a.n_cols()));
  out << a.eigen(), b.eigen();
  return Mat<T>(std::move(out));
}

template <typename T>
inline T sum(const Mat<T>& x) {
  return x.eigen().sum();
}

template <typename T>
inline T sum(const Col<T>& x) {
  return x.eigen().sum();
}

template <typename T>
inline T sum(const Row<T>& x) {
  return x.eigen().sum();
}

template <typename T>
inline T accu(const Mat<T>& x) {
  return sum(x);
}

template <typename T>
inline T mean(const Mat<T>& x) {
  if (x.n_elem() == 0) {
    return T(0);
  }
  return sum(x) / static_cast<T>(x.n_elem());
}

template <typename T>
inline T min(const Mat<T>& x) {
  DINGO_ASSERT(x.n_elem() > 0, "dingo::min on empty matrix");
  return x.eigen().minCoeff();
}

template <typename T>
inline T max(const Mat<T>& x) {
  DINGO_ASSERT(x.n_elem() > 0, "dingo::max on empty matrix");
  return x.eigen().maxCoeff();
}

template <typename T>
inline T var(const Mat<T>& x) {
  if (x.n_elem() < 2) {
    return T(0);
  }
  const auto m = mean(x);
  const auto centered = x.eigen().array() - m;
  return centered.square().sum() / static_cast<T>(x.n_elem() - 1);
}

template <typename T>
inline T stddev(const Mat<T>& x) {
  return std::sqrt(var(x));
}

template <typename T>
inline T norm(const Mat<T>& x, int p = 2) {
  if (p == 2) {
    return x.eigen().norm();
  }
  if (p == 1) {
    return x.eigen().template lpNorm<1>();
  }
  if (p == 0) {
    return static_cast<T>((x.eigen().array() != T(0)).count());
  }
  throw std::invalid_argument("dingo::norm supports p=0,1,2 only in v1");
}

template <typename T>
inline T trace(const Mat<T>& x) {
  return x.eigen().trace();
}

}  // namespace dingo
