#pragma once

#include <algorithm>
#include <complex>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

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

  class SubmatProxy {
   public:
    SubmatProxy(Mat& parent, uword row1, uword col1, uword row2, uword col2)
        : parent_(parent), row1_(row1), col1_(col1), row2_(row2), col2_(col2) {}

    SubmatProxy& operator=(const Mat& rhs) {
      const auto rows = row2_ - row1_ + 1;
      const auto cols = col2_ - col1_ + 1;
      DINGO_ASSERT(rhs.n_rows() == rows && rhs.n_cols() == cols, "dingo::Mat subview size mismatch in assignment");
      parent_.data_.block(static_cast<Eigen::Index>(row1_), static_cast<Eigen::Index>(col1_),
                          static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)) = rhs.data_;
      return *this;
    }

    SubmatProxy& operator=(const T& value) {
      const auto rows = row2_ - row1_ + 1;
      const auto cols = col2_ - col1_ + 1;
      parent_.data_.block(static_cast<Eigen::Index>(row1_), static_cast<Eigen::Index>(col1_),
                          static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols))
          .setConstant(value);
      return *this;
    }

    [[nodiscard]] Mat eval() const { return parent_.submat(row1_, col1_, row2_, col2_); }
    operator Mat() const { return eval(); }

   private:
    Mat& parent_;
    uword row1_{0};
    uword col1_{0};
    uword row2_{0};
    uword col2_{0};
  };

  template <typename U>
  class MaskProxy {
   public:
    MaskProxy(Mat& parent, const Mat<U>& mask) : parent_(parent), mask_(mask) {
      DINGO_ASSERT(mask_.n_rows() == parent_.n_rows() && mask_.n_cols() == parent_.n_cols(),
                   "dingo::Mat mask size mismatch");
    }

    MaskProxy& operator=(const T& value) {
      for (uword i = 0; i < parent_.n_elem(); ++i) {
        if (mask_(i) != U(0)) {
          parent_(i) = value;
        }
      }
      return *this;
    }

    MaskProxy& operator=(const Mat<T>& values) {
      const auto idx = selected_indices();
      DINGO_ASSERT(values.n_elem() == idx.size(), "dingo::Mat mask assignment value count mismatch");
      for (uword i = 0; i < idx.size(); ++i) {
        parent_(idx[i]) = values(i);
      }
      return *this;
    }

    [[nodiscard]] Mat eval() const {
      const auto idx = selected_indices();
      Mat out(idx.size(), 1, fill::zeros);
      for (uword i = 0; i < idx.size(); ++i) {
        out(i, 0) = parent_(idx[i]);
      }
      return out;
    }

    operator Mat() const { return eval(); }

   private:
    [[nodiscard]] std::vector<uword> selected_indices() const {
      std::vector<uword> idx;
      idx.reserve(parent_.n_elem());
      for (uword i = 0; i < parent_.n_elem(); ++i) {
        if (mask_(i) != U(0)) {
          idx.push_back(i);
        }
      }
      return idx;
    }

    Mat& parent_;
    const Mat<U>& mask_;
  };

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

  [[nodiscard]] Mat submat(uword row1, uword col1, uword row2, uword col2) const {
    DINGO_ASSERT(row1 <= row2 && col1 <= col2, "dingo::Mat invalid submat bounds");
    DINGO_ASSERT(row2 < n_rows() && col2 < n_cols(), "dingo::Mat submat index out of range");
    const auto rows = static_cast<Eigen::Index>(row2 - row1 + 1);
    const auto cols = static_cast<Eigen::Index>(col2 - col1 + 1);
    storage_type out = data_.block(static_cast<Eigen::Index>(row1), static_cast<Eigen::Index>(col1), rows, cols);
    return Mat(std::move(out));
  }

  [[nodiscard]] Mat submat(span rows, span cols) const {
    const auto r1 = rows.begin;
    const auto c1 = cols.begin;
    const auto r2 = (rows.end == span_all_end()) ? (n_rows() - 1) : rows.end;
    const auto c2 = (cols.end == span_all_end()) ? (n_cols() - 1) : cols.end;
    return submat(r1, c1, r2, c2);
  }

  [[nodiscard]] SubmatProxy operator()(span rows, span cols) {
    const auto r1 = rows.begin;
    const auto c1 = cols.begin;
    const auto r2 = (rows.end == span_all_end()) ? (n_rows() - 1) : rows.end;
    const auto c2 = (cols.end == span_all_end()) ? (n_cols() - 1) : cols.end;
    DINGO_ASSERT(r1 <= r2 && c1 <= c2, "dingo::Mat invalid subview bounds");
    DINGO_ASSERT(r2 < n_rows() && c2 < n_cols(), "dingo::Mat subview index out of range");
    return SubmatProxy(*this, r1, c1, r2, c2);
  }

  [[nodiscard]] Mat operator()(span rows, span cols) const {
    return submat(rows, cols);
  }

  [[nodiscard]] Mat submat(all_t, span cols) const { return submat(span{0, span_all_end()}, cols); }
  [[nodiscard]] Mat submat(span rows, all_t) const { return submat(rows, span{0, span_all_end()}); }
  [[nodiscard]] Mat submat(all_t, all_t) const { return *this; }

  [[nodiscard]] SubmatProxy operator()(all_t, span cols) { return (*this)(span{0, span_all_end()}, cols); }
  [[nodiscard]] SubmatProxy operator()(span rows, all_t) { return (*this)(rows, span{0, span_all_end()}); }
  [[nodiscard]] SubmatProxy operator()(all_t, all_t) { return (*this)(span{0, span_all_end()}, span{0, span_all_end()}); }
  [[nodiscard]] Mat operator()(all_t, span cols) const { return submat(all_t{}, cols); }
  [[nodiscard]] Mat operator()(span rows, all_t) const { return submat(rows, all_t{}); }
  [[nodiscard]] Mat operator()(all_t, all_t) const { return *this; }

  template <typename U>
  [[nodiscard]] MaskProxy<U> operator()(const Mat<U>& mask) {
    return MaskProxy<U>(*this, mask);
  }

  template <typename U>
  [[nodiscard]] Mat operator()(const Mat<U>& mask) const {
    DINGO_ASSERT(mask.n_rows() == n_rows() && mask.n_cols() == n_cols(), "dingo::Mat mask size mismatch");
    std::vector<uword> idx;
    idx.reserve(n_elem());
    for (uword i = 0; i < n_elem(); ++i) {
      if (mask(i) != U(0)) {
        idx.push_back(i);
      }
    }
    Mat out(idx.size(), 1, fill::zeros);
    for (uword i = 0; i < idx.size(); ++i) {
      out(i, 0) = (*this)(idx[i]);
    }
    return out;
  }

  void set_submat(uword row1, uword col1, const Mat& block) {
    DINGO_ASSERT(row1 + block.n_rows() <= n_rows() && col1 + block.n_cols() <= n_cols(),
                 "dingo::Mat set_submat out of range");
    data_.block(static_cast<Eigen::Index>(row1), static_cast<Eigen::Index>(col1),
                static_cast<Eigen::Index>(block.n_rows()), static_cast<Eigen::Index>(block.n_cols())) = block.data_;
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

template <typename T>
inline Mat<T> operator+(const Mat<T>& lhs, const Col<T>& rhs) {
  DINGO_ASSERT(lhs.n_rows() == rhs.n_rows(), "dingo::Mat/Col row mismatch for +");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().colwise() + rhs.eigen()).eval()));
}

template <typename T>
inline Mat<T> operator+(const Mat<T>& lhs, const Row<T>& rhs) {
  DINGO_ASSERT(lhs.n_cols() == rhs.n_cols(), "dingo::Mat/Row col mismatch for +");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().rowwise() + rhs.eigen()).eval()));
}

template <typename T>
inline Mat<T> operator-(const Mat<T>& lhs, const Col<T>& rhs) {
  DINGO_ASSERT(lhs.n_rows() == rhs.n_rows(), "dingo::Mat/Col row mismatch for -");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().colwise() - rhs.eigen()).eval()));
}

template <typename T>
inline Mat<T> operator-(const Mat<T>& lhs, const Row<T>& rhs) {
  DINGO_ASSERT(lhs.n_cols() == rhs.n_cols(), "dingo::Mat/Row col mismatch for -");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().rowwise() - rhs.eigen()).eval()));
}

template <typename T>
inline Mat<T> operator%(const Mat<T>& lhs, const Col<T>& rhs) {
  DINGO_ASSERT(lhs.n_rows() == rhs.n_rows(), "dingo::Mat/Col row mismatch for %");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().array().colwise() * rhs.eigen().array()).matrix()));
}

template <typename T>
inline Mat<T> operator%(const Mat<T>& lhs, const Row<T>& rhs) {
  DINGO_ASSERT(lhs.n_cols() == rhs.n_cols(), "dingo::Mat/Row col mismatch for %");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().array().rowwise() * rhs.eigen().array()).matrix()));
}

template <typename T>
inline Mat<T> operator/(const Mat<T>& lhs, const Col<T>& rhs) {
  DINGO_ASSERT(lhs.n_rows() == rhs.n_rows(), "dingo::Mat/Col row mismatch for /");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().array().colwise() / rhs.eigen().array()).matrix()));
}

template <typename T>
inline Mat<T> operator/(const Mat<T>& lhs, const Row<T>& rhs) {
  DINGO_ASSERT(lhs.n_cols() == rhs.n_cols(), "dingo::Mat/Row col mismatch for /");
  return Mat<T>(typename Mat<T>::storage_type((lhs.eigen().array().rowwise() / rhs.eigen().array()).matrix()));
}

using mat = Mat<double>;
using cx_mat = Mat<std::complex<double>>;

using vec = Col<double>;
using cx_vec = Col<std::complex<double>>;
using colvec = vec;
using cx_colvec = cx_vec;

using rowvec = Row<double>;
using cx_rowvec = Row<std::complex<double>>;

}  // namespace dingo
