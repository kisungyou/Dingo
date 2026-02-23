#pragma once

#include <vector>

#include "fill.hpp"
#include "matrix.hpp"
#include "../detail/assert.hpp"

namespace dingo {

template <typename T>
class Cube {
 public:
  using value_type = T;

  Cube() = default;

  Cube(uword rows, uword cols, uword slices)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), slices_(slices, Mat<T>(rows, cols, fill::zeros)) {}

  Cube(uword rows, uword cols, uword slices, fill::zeros_t)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), slices_(slices, Mat<T>(rows, cols, fill::zeros)) {}

  Cube(uword rows, uword cols, uword slices, fill::ones_t)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), slices_(slices, Mat<T>(rows, cols, fill::ones)) {}

  Cube(uword rows, uword cols, uword slices, fill::randu_t)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), slices_(slices, Mat<T>(rows, cols, fill::zeros)) {
    randu();
  }

  Cube(uword rows, uword cols, uword slices, fill::randn_t)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), slices_(slices, Mat<T>(rows, cols, fill::zeros)) {
    randn();
  }

  [[nodiscard]] uword n_rows() const { return n_rows_; }
  [[nodiscard]] uword n_cols() const { return n_cols_; }
  [[nodiscard]] uword n_slices() const { return n_slices_; }
  [[nodiscard]] uword n_elem() const { return n_rows_ * n_cols_ * n_slices_; }

  void resize(uword rows, uword cols, uword slices) {
    n_rows_ = rows;
    n_cols_ = cols;
    n_slices_ = slices;
    slices_.assign(slices, Mat<T>(rows, cols, fill::zeros));
  }

  T& operator()(uword row, uword col, uword slice) {
    DINGO_ASSERT(row < n_rows_ && col < n_cols_ && slice < n_slices_, "dingo::Cube index out of range");
    return slices_[slice](row, col);
  }

  const T& operator()(uword row, uword col, uword slice) const {
    DINGO_ASSERT(row < n_rows_ && col < n_cols_ && slice < n_slices_, "dingo::Cube index out of range");
    return slices_[slice](row, col);
  }

  Mat<T>& slice(uword idx) {
    DINGO_ASSERT(idx < n_slices_, "dingo::Cube slice index out of range");
    return slices_[idx];
  }

  const Mat<T>& slice(uword idx) const {
    DINGO_ASSERT(idx < n_slices_, "dingo::Cube slice index out of range");
    return slices_[idx];
  }

  void zeros() {
    for (auto& s : slices_) {
      s.zeros();
    }
  }

  void ones() {
    for (auto& s : slices_) {
      s.ones();
    }
  }

  void randu() {
    for (auto& s : slices_) {
      s.randu();
    }
  }

  void randn() {
    for (auto& s : slices_) {
      s.randn();
    }
  }

 private:
  uword n_rows_{0};
  uword n_cols_{0};
  uword n_slices_{0};
  std::vector<Mat<T>> slices_;
};

using cube = Cube<double>;
using cx_cube = Cube<std::complex<double>>;

}  // namespace dingo
