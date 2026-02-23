#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cube.hpp"
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
inline Mat<T> sum(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::sum dim must be 0 or 1");
  if (dim == 0) {
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    out.row(0) = x.eigen().colwise().sum();
    return Mat<T>(std::move(out));
  }
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  out.col(0) = x.eigen().rowwise().sum();
  return Mat<T>(std::move(out));
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
inline Mat<T> mean(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::mean dim must be 0 or 1");
  if (dim == 0) {
    DINGO_ASSERT(x.n_rows() > 0, "dingo::mean dim=0 on empty matrix");
    return sum(x, 0) / static_cast<T>(x.n_rows());
  }
  DINGO_ASSERT(x.n_cols() > 0, "dingo::mean dim=1 on empty matrix");
  return sum(x, 1) / static_cast<T>(x.n_cols());
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

template <typename T>
inline Col<T> diagvec(const Mat<T>& x) {
  const auto n = static_cast<Eigen::Index>(std::min(x.n_rows(), x.n_cols()));
  typename Col<T>::storage_type out(n);
  out = x.eigen().diagonal();
  return Col<T>(std::move(out));
}

template <typename T>
inline Mat<T> diagmat(const Mat<T>& x) {
  typename Mat<T>::storage_type out =
      Mat<T>::storage_type::Zero(static_cast<Eigen::Index>(x.n_rows()), static_cast<Eigen::Index>(x.n_cols()));
  const auto n = static_cast<Eigen::Index>(std::min(x.n_rows(), x.n_cols()));
  out.diagonal().head(n) = x.eigen().diagonal().head(n);
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> diagmat(const Col<T>& v) {
  typename Mat<T>::storage_type out =
      Mat<T>::storage_type::Zero(static_cast<Eigen::Index>(v.n_elem()), static_cast<Eigen::Index>(v.n_elem()));
  out.diagonal() = v.eigen();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> tril(const Mat<T>& x, int k = 0) {
  typename Mat<T>::storage_type out =
      Mat<T>::storage_type::Zero(static_cast<Eigen::Index>(x.n_rows()), static_cast<Eigen::Index>(x.n_cols()));
  for (uword r = 0; r < x.n_rows(); ++r) {
    for (uword c = 0; c < x.n_cols(); ++c) {
      if (static_cast<int>(r) - static_cast<int>(c) >= -k) {
        out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = x(r, c);
      }
    }
  }
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> triu(const Mat<T>& x, int k = 0) {
  typename Mat<T>::storage_type out =
      Mat<T>::storage_type::Zero(static_cast<Eigen::Index>(x.n_rows()), static_cast<Eigen::Index>(x.n_cols()));
  for (uword r = 0; r < x.n_rows(); ++r) {
    for (uword c = 0; c < x.n_cols(); ++c) {
      if (static_cast<int>(c) - static_cast<int>(r) >= k) {
        out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = x(r, c);
      }
    }
  }
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> sin(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().unaryExpr([](const T& v) {
    using std::sin;
    return sin(v);
  });
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> cos(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().unaryExpr([](const T& v) {
    using std::cos;
    return cos(v);
  });
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> exp(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().unaryExpr([](const T& v) {
    using std::exp;
    return exp(v);
  });
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> log(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().unaryExpr([](const T& v) {
    using std::log;
    return log(v);
  });
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> sqrt(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().unaryExpr([](const T& v) {
    using std::sqrt;
    return sqrt(v);
  });
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<typename Eigen::NumTraits<T>::Real> abs(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  typename Mat<real_t>::storage_type out = x.eigen().unaryExpr([](const T& v) -> real_t {
    using std::abs;
    return static_cast<real_t>(abs(v));
  });
  return Mat<real_t>(std::move(out));
}

template <typename T>
inline Mat<T> conj(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().conjugate();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<typename Eigen::NumTraits<T>::Real> real(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  typename Mat<real_t>::storage_type out = x.eigen().real();
  return Mat<real_t>(std::move(out));
}

template <typename T>
inline Mat<typename Eigen::NumTraits<T>::Real> imag(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  typename Mat<real_t>::storage_type out = x.eigen().imag();
  return Mat<real_t>(std::move(out));
}

template <typename T>
inline Mat<typename Eigen::NumTraits<T>::Real> angle(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  typename Mat<real_t>::storage_type out = x.eigen().unaryExpr([](const T& v) -> real_t {
    using std::arg;
    return static_cast<real_t>(arg(v));
  });
  return Mat<real_t>(std::move(out));
}

template <typename T>
inline Mat<typename Eigen::NumTraits<T>::Real> abs2(const Mat<T>& x) {
  using real_t = typename Eigen::NumTraits<T>::Real;
  typename Mat<real_t>::storage_type out = x.eigen().cwiseAbs2();
  return Mat<real_t>(std::move(out));
}

template <typename T>
inline Mat<T> min(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::min dim must be 0 or 1");
  if (dim == 0) {
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    out.row(0) = x.eigen().colwise().minCoeff();
    return Mat<T>(std::move(out));
  }
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  out.col(0) = x.eigen().rowwise().minCoeff();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> max(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::max dim must be 0 or 1");
  if (dim == 0) {
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    out.row(0) = x.eigen().colwise().maxCoeff();
    return Mat<T>(std::move(out));
  }
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  out.col(0) = x.eigen().rowwise().maxCoeff();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> var(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::var dim must be 0 or 1");
  if (dim == 0) {
    DINGO_ASSERT(x.n_rows() > 1, "dingo::var dim=0 requires at least 2 rows");
    const auto m = mean(x, 0);
    typename Mat<T>::storage_type centered = x.eigen().rowwise() - m.eigen().row(0);
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    out.row(0) = centered.array().square().colwise().sum() / static_cast<T>(x.n_rows() - 1);
    return Mat<T>(std::move(out));
  }
  DINGO_ASSERT(x.n_cols() > 1, "dingo::var dim=1 requires at least 2 cols");
  const auto m = mean(x, 1);
  typename Mat<T>::storage_type centered = x.eigen().colwise() - m.eigen().col(0);
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  out.col(0) = centered.array().square().rowwise().sum() / static_cast<T>(x.n_cols() - 1);
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> stddev(const Mat<T>& x, uword dim) {
  return sqrt(var(x, dim));
}

template <typename T>
inline Mat<T> any(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::any dim must be 0 or 1");
  if (dim == 0) {
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    for (uword j = 0; j < x.n_cols(); ++j) {
      out(0, static_cast<Eigen::Index>(j)) = (x.eigen().col(static_cast<Eigen::Index>(j)).array() != T(0)).any() ? T(1) : T(0);
    }
    return Mat<T>(std::move(out));
  }
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  for (uword i = 0; i < x.n_rows(); ++i) {
    out(static_cast<Eigen::Index>(i), 0) = (x.eigen().row(static_cast<Eigen::Index>(i)).array() != T(0)).any() ? T(1) : T(0);
  }
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> all(const Mat<T>& x, uword dim) {
  DINGO_ASSERT(dim <= 1, "dingo::all dim must be 0 or 1");
  if (dim == 0) {
    typename Mat<T>::storage_type out(1, static_cast<Eigen::Index>(x.n_cols()));
    for (uword j = 0; j < x.n_cols(); ++j) {
      out(0, static_cast<Eigen::Index>(j)) = (x.eigen().col(static_cast<Eigen::Index>(j)).array() != T(0)).all() ? T(1) : T(0);
    }
    return Mat<T>(std::move(out));
  }
  typename Mat<T>::storage_type out(static_cast<Eigen::Index>(x.n_rows()), 1);
  for (uword i = 0; i < x.n_rows(); ++i) {
    out(static_cast<Eigen::Index>(i), 0) = (x.eigen().row(static_cast<Eigen::Index>(i)).array() != T(0)).all() ? T(1) : T(0);
  }
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> flipud(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().colwise().reverse();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> fliplr(const Mat<T>& x) {
  typename Mat<T>::storage_type out = x.eigen().rowwise().reverse();
  return Mat<T>(std::move(out));
}

template <typename T>
inline Mat<T> repmat(const Mat<T>& x, uword row_reps, uword col_reps) {
  typename Mat<T>::storage_type out = x.eigen().replicate(static_cast<Eigen::Index>(row_reps), static_cast<Eigen::Index>(col_reps));
  return Mat<T>(std::move(out));
}

template <typename T>
inline Col<T> sort(const Col<T>& x) {
  std::vector<T> values(static_cast<std::size_t>(x.n_elem()));
  for (uword i = 0; i < x.n_elem(); ++i) values[i] = x(i);
  std::sort(values.begin(), values.end());
  Col<T> out(x.n_elem());
  for (uword i = 0; i < x.n_elem(); ++i) out(i) = values[i];
  return out;
}

template <typename T>
inline Row<T> sort(const Row<T>& x) {
  std::vector<T> values(static_cast<std::size_t>(x.n_elem()));
  for (uword i = 0; i < x.n_elem(); ++i) values[i] = x(i);
  std::sort(values.begin(), values.end());
  Row<T> out(x.n_elem());
  for (uword i = 0; i < x.n_elem(); ++i) out(i) = values[i];
  return out;
}

template <typename T>
inline Col<T> unique(const Col<T>& x) {
  auto sorted = sort(x);
  std::vector<T> vals;
  vals.reserve(static_cast<std::size_t>(x.n_elem()));
  for (uword i = 0; i < sorted.n_elem(); ++i) {
    if (vals.empty() || vals.back() != sorted(i)) vals.push_back(sorted(i));
  }
  Col<T> out(vals.size());
  for (uword i = 0; i < vals.size(); ++i) out(i) = vals[i];
  return out;
}

template <typename T>
inline Col<uword> find(const Mat<T>& x) {
  std::vector<uword> idx;
  idx.reserve(static_cast<std::size_t>(x.n_elem()));
  for (uword i = 0; i < x.n_elem(); ++i) {
    if (x(i) != T(0)) idx.push_back(i);
  }
  Col<uword> out(idx.size());
  for (uword i = 0; i < idx.size(); ++i) out(i) = idx[i];
  return out;
}

template <typename T>
inline Col<T> elem(const Mat<T>& x, const Col<uword>& idx) {
  Col<T> out(idx.n_elem());
  for (uword i = 0; i < idx.n_elem(); ++i) {
    DINGO_ASSERT(idx(i) < x.n_elem(), "dingo::elem index out of range");
    out(i) = x(idx(i));
  }
  return out;
}

template <typename T>
inline Mat<T> squeeze(const Cube<T>& x) {
  DINGO_ASSERT(x.n_slices() == 1, "dingo::squeeze currently supports n_slices==1");
  return x.slice(0);
}

template <typename T>
inline Cube<T> permute(const Cube<T>& x, uword d1, uword d2, uword d3) {
  DINGO_ASSERT(d1 >= 1 && d1 <= 3 && d2 >= 1 && d2 <= 3 && d3 >= 1 && d3 <= 3, "dingo::permute dims must be 1..3");
  DINGO_ASSERT(d1 != d2 && d1 != d3 && d2 != d3, "dingo::permute dims must be unique");
  const uword dims[3] = {x.n_rows(), x.n_cols(), x.n_slices()};
  const uword new_dims[3] = {dims[d1 - 1], dims[d2 - 1], dims[d3 - 1]};
  Cube<T> out(new_dims[0], new_dims[1], new_dims[2], fill::zeros);
  for (uword i = 0; i < x.n_rows(); ++i) {
    for (uword j = 0; j < x.n_cols(); ++j) {
      for (uword k = 0; k < x.n_slices(); ++k) {
        const uword old_pos[3] = {i, j, k};
        out(old_pos[d1 - 1], old_pos[d2 - 1], old_pos[d3 - 1]) = x(i, j, k);
      }
    }
  }
  return out;
}

template <typename T>
inline Cube<T> join_slices(const Cube<T>& a, const Cube<T>& b) {
  DINGO_ASSERT(a.n_rows() == b.n_rows() && a.n_cols() == b.n_cols(), "dingo::join_slices size mismatch");
  Cube<T> out(a.n_rows(), a.n_cols(), a.n_slices() + b.n_slices(), fill::zeros);
  for (uword s = 0; s < a.n_slices(); ++s) {
    out.slice(s) = a.slice(s);
  }
  for (uword s = 0; s < b.n_slices(); ++s) {
    out.slice(a.n_slices() + s) = b.slice(s);
  }
  return out;
}

}  // namespace dingo
