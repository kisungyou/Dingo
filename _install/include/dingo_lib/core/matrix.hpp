#pragma once

#include <algorithm>
#include <complex>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "fill.hpp"
#include "types.hpp"
#include "../detail/assert.hpp"
#include "../detail/eigen_dense.hpp"
#include "../detail/random.hpp"

namespace dingo {

template <typename T>
class Mat {
 public:
  using value_type = T;
  using storage_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  Mat() = default;

  explicit Mat(uword n_elem) : data_(static_cast<Eigen::Index>(n_elem), 1) { data_.setZero(); }

  Mat(uword rows, uword cols)
      : data_(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)) {
    data_.setZero();
  }

  Mat(uword rows, uword cols, fill::zeros_t)
      : data_(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)) {
    data_.setZero();
  }

  Mat(uword rows, uword cols, fill::ones_t)
      : data_(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)) {
    data_.setOnes();
  }

  Mat(uword rows, uword cols, fill::randu_t)
      : data_(make_random(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols), true)) {}

  Mat(uword rows, uword cols, fill::randn_t)
      : data_(make_random(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols), false)) {}

  Mat(std::initializer_list<std::initializer_list<T>> values) {
    const auto rows = static_cast<Eigen::Index>(values.size());
    Eigen::Index cols = 0;
    for (const auto& row_values : values) {
      cols = std::max(cols, static_cast<Eigen::Index>(row_values.size()));
    }
    data_.resize(rows, cols);
    data_.setZero();

    Eigen::Index r = 0;
    for (const auto& row_values : values) {
      Eigen::Index c = 0;
      for (const auto& value : row_values) {
        data_(r, c) = value;
        ++c;
      }
      ++r;
    }
  }

  explicit Mat(const storage_type& data) : data_(data) {}
  explicit Mat(storage_type&& data) : data_(std::move(data)) {}

  [[nodiscard]] uword n_rows() const { return static_cast<uword>(data_.rows()); }
  [[nodiscard]] uword n_cols() const { return static_cast<uword>(data_.cols()); }
  [[nodiscard]] uword n_elem() const { return static_cast<uword>(data_.size()); }
  [[nodiscard]] bool is_empty() const { return data_.size() == 0; }

  void resize(uword rows, uword cols) {
    data_.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  }

  void zeros() { data_.setZero(); }
  void ones() { data_.setOnes(); }

  void randu() {
    data_ = make_random(data_.rows(), data_.cols(), true);
  }

  void randn() {
    data_ = make_random(data_.rows(), data_.cols(), false);
  }

  T& operator()(uword row, uword col) {
    DINGO_ASSERT(row < n_rows() && col < n_cols(), "dingo::Mat index out of range");
    return data_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }

  const T& operator()(uword row, uword col) const {
    DINGO_ASSERT(row < n_rows() && col < n_cols(), "dingo::Mat index out of range");
    return data_(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
  }

  T& operator()(uword index) {
    DINGO_ASSERT(index < n_elem(), "dingo::Mat linear index out of range");
    return data_(static_cast<Eigen::Index>(index));
  }

  const T& operator()(uword index) const {
    DINGO_ASSERT(index < n_elem(), "dingo::Mat linear index out of range");
    return data_(static_cast<Eigen::Index>(index));
  }

  T& at(uword row, uword col) { return (*this)(row, col); }
  const T& at(uword row, uword col) const { return (*this)(row, col); }

  [[nodiscard]] Mat row(uword idx) const {
    DINGO_ASSERT(idx < n_rows(), "dingo::Mat row index out of range");
    storage_type out = data_.row(static_cast<Eigen::Index>(idx));
    return Mat(std::move(out));
  }

  [[nodiscard]] Mat col(uword idx) const {
    DINGO_ASSERT(idx < n_cols(), "dingo::Mat col index out of range");
    storage_type out = data_.col(static_cast<Eigen::Index>(idx));
    return Mat(std::move(out));
  }

  void set_row(uword idx, const Mat& values) {
    DINGO_ASSERT(idx < n_rows(), "dingo::Mat row index out of range");
    DINGO_ASSERT(values.n_rows() == 1 && values.n_cols() == n_cols(), "dingo::Mat row size mismatch");
    data_.row(static_cast<Eigen::Index>(idx)) = values.data_;
  }

  void set_col(uword idx, const Mat& values) {
    DINGO_ASSERT(idx < n_cols(), "dingo::Mat col index out of range");
    DINGO_ASSERT(values.n_cols() == 1 && values.n_rows() == n_rows(), "dingo::Mat col size mismatch");
    data_.col(static_cast<Eigen::Index>(idx)) = values.data_;
  }

  [[nodiscard]] Mat t() const { return Mat(storage_type(data_.transpose())); }
  [[nodiscard]] Mat st() const { return Mat(storage_type(data_.adjoint())); }

  storage_type& eigen() { return data_; }
  const storage_type& eigen() const { return data_; }

  Mat& operator+=(const Mat& rhs) {
    DINGO_ASSERT(n_rows() == rhs.n_rows() && n_cols() == rhs.n_cols(), "dingo::Mat size mismatch for +=");
    data_ += rhs.data_;
    return *this;
  }

  Mat& operator-=(const Mat& rhs) {
    DINGO_ASSERT(n_rows() == rhs.n_rows() && n_cols() == rhs.n_cols(), "dingo::Mat size mismatch for -=");
    data_ -= rhs.data_;
    return *this;
  }

  Mat& operator%=(const Mat& rhs) {
    DINGO_ASSERT(n_rows() == rhs.n_rows() && n_cols() == rhs.n_cols(), "dingo::Mat size mismatch for %=");
    data_ = (data_.array() * rhs.data_.array()).matrix();
    return *this;
  }

  Mat& operator*=(const T& scalar) {
    data_ *= scalar;
    return *this;
  }

  Mat& operator/=(const T& scalar) {
    data_ /= scalar;
    return *this;
  }

  [[nodiscard]] Mat operator-() const { return Mat(-data_); }

 private:
  static storage_type make_random(Eigen::Index rows, Eigen::Index cols, bool uniform) {
    storage_type out(rows, cols);
    out = storage_type::NullaryExpr(rows, cols, [uniform]() -> T {
      return uniform ? detail::randu<T>() : detail::randn<T>();
    });
    return out;
  }

  storage_type data_;
};

template <typename T>
inline Mat<T> operator+(Mat<T> lhs, const Mat<T>& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T>
inline Mat<T> operator-(Mat<T> lhs, const Mat<T>& rhs) {
  lhs -= rhs;
  return lhs;
}

template <typename T>
inline Mat<T> operator%(Mat<T> lhs, const Mat<T>& rhs) {
  lhs %= rhs;
  return lhs;
}

template <typename T>
inline Mat<T> operator*(const Mat<T>& lhs, const Mat<T>& rhs) {
  DINGO_ASSERT(lhs.n_cols() == rhs.n_rows(), "dingo::Mat size mismatch for *");
  return Mat<T>(typename Mat<T>::storage_type(lhs.eigen() * rhs.eigen()));
}

template <typename T>
inline Mat<T> operator*(Mat<T> lhs, const T& scalar) {
  lhs *= scalar;
  return lhs;
}

template <typename T>
inline Mat<T> operator*(const T& scalar, Mat<T> rhs) {
  rhs *= scalar;
  return rhs;
}

template <typename T>
inline Mat<T> operator/(Mat<T> lhs, const T& scalar) {
  lhs /= scalar;
  return lhs;
}

template <typename T>
inline Mat<T> operator/(const Mat<T>& lhs, const Mat<T>& rhs) {
  DINGO_ASSERT(lhs.n_rows() == rhs.n_rows() && lhs.n_cols() == rhs.n_cols(), "dingo::Mat size mismatch for /");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().array() / rhs.eigen().array()).matrix()));
}

template <typename T>
class Col {
 public:
  using value_type = T;
  using storage_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Col() = default;

  explicit Col(uword n_elem) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setZero(); }
  Col(uword n_elem, fill::zeros_t) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setZero(); }
  Col(uword n_elem, fill::ones_t) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setOnes(); }
  Col(uword n_elem, fill::randu_t) : data_(make_random(static_cast<Eigen::Index>(n_elem), true)) {}
  Col(uword n_elem, fill::randn_t) : data_(make_random(static_cast<Eigen::Index>(n_elem), false)) {}

  Col(std::initializer_list<T> values) : data_(static_cast<Eigen::Index>(values.size())) {
    Eigen::Index i = 0;
    for (const auto& value : values) {
      data_(i++) = value;
    }
  }

  explicit Col(const storage_type& data) : data_(data) {}
  explicit Col(storage_type&& data) : data_(std::move(data)) {}

  [[nodiscard]] uword n_rows() const { return static_cast<uword>(data_.rows()); }
  [[nodiscard]] uword n_cols() const { return 1; }
  [[nodiscard]] uword n_elem() const { return static_cast<uword>(data_.size()); }

  T& operator()(uword i) {
    DINGO_ASSERT(i < n_elem(), "dingo::Col index out of range");
    return data_(static_cast<Eigen::Index>(i));
  }

  const T& operator()(uword i) const {
    DINGO_ASSERT(i < n_elem(), "dingo::Col index out of range");
    return data_(static_cast<Eigen::Index>(i));
  }

  void zeros() { data_.setZero(); }
  void ones() { data_.setOnes(); }
  void randu() { data_ = make_random(data_.rows(), true); }
  void randn() { data_ = make_random(data_.rows(), false); }

  storage_type& eigen() { return data_; }
  const storage_type& eigen() const { return data_; }

  [[nodiscard]] Mat<T> as_mat() const {
    typename Mat<T>::storage_type out = data_;
    return Mat<T>(std::move(out));
  }

 private:
  static storage_type make_random(Eigen::Index rows, bool uniform) {
    storage_type out(rows);
    out = storage_type::NullaryExpr(rows, [uniform]() -> T { return uniform ? detail::randu<T>() : detail::randn<T>(); });
    return out;
  }

  storage_type data_;
};

template <typename T>
class Row {
 public:
  using value_type = T;
  using storage_type = Eigen::Matrix<T, 1, Eigen::Dynamic>;

  Row() = default;

  explicit Row(uword n_elem) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setZero(); }
  Row(uword n_elem, fill::zeros_t) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setZero(); }
  Row(uword n_elem, fill::ones_t) : data_(static_cast<Eigen::Index>(n_elem)) { data_.setOnes(); }
  Row(uword n_elem, fill::randu_t) : data_(make_random(static_cast<Eigen::Index>(n_elem), true)) {}
  Row(uword n_elem, fill::randn_t) : data_(make_random(static_cast<Eigen::Index>(n_elem), false)) {}

  Row(std::initializer_list<T> values) : data_(static_cast<Eigen::Index>(values.size())) {
    Eigen::Index i = 0;
    for (const auto& value : values) {
      data_(i++) = value;
    }
  }

  explicit Row(const storage_type& data) : data_(data) {}
  explicit Row(storage_type&& data) : data_(std::move(data)) {}

  [[nodiscard]] uword n_rows() const { return 1; }
  [[nodiscard]] uword n_cols() const { return static_cast<uword>(data_.cols()); }
  [[nodiscard]] uword n_elem() const { return static_cast<uword>(data_.size()); }

  T& operator()(uword i) {
    DINGO_ASSERT(i < n_elem(), "dingo::Row index out of range");
    return data_(static_cast<Eigen::Index>(i));
  }

  const T& operator()(uword i) const {
    DINGO_ASSERT(i < n_elem(), "dingo::Row index out of range");
    return data_(static_cast<Eigen::Index>(i));
  }

  void zeros() { data_.setZero(); }
  void ones() { data_.setOnes(); }
  void randu() { data_ = make_random(data_.cols(), true); }
  void randn() { data_ = make_random(data_.cols(), false); }

  storage_type& eigen() { return data_; }
  const storage_type& eigen() const { return data_; }

  [[nodiscard]] Mat<T> as_mat() const {
    typename Mat<T>::storage_type out = data_;
    return Mat<T>(std::move(out));
  }

 private:
  static storage_type make_random(Eigen::Index cols, bool uniform) {
    storage_type out(cols);
    out = storage_type::NullaryExpr(cols, [uniform]() -> T { return uniform ? detail::randu<T>() : detail::randn<T>(); });
    return out;
  }

  storage_type data_;
};

using mat = Mat<double>;
using fmat = Mat<float>;
using cx_mat = Mat<std::complex<double>>;
using cx_fmat = Mat<std::complex<float>>;

using vec = Col<double>;
using fvec = Col<float>;
using cx_vec = Col<std::complex<double>>;
using cx_fvec = Col<std::complex<float>>;

using rowvec = Row<double>;
using frowvec = Row<float>;
using cx_rowvec = Row<std::complex<double>>;
using cx_frowvec = Row<std::complex<float>>;

}  // namespace dingo
