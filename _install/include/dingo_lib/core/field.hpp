#pragma once

#include <vector>

#include "types.hpp"
#include "../detail/assert.hpp"

namespace dingo {

template <typename T>
class Field {
 public:
  Field() = default;

  explicit Field(uword n_elem) : n_rows_(n_elem), n_cols_(1), n_slices_(1), data_(n_elem) {}

  Field(uword rows, uword cols) : n_rows_(rows), n_cols_(cols), n_slices_(1), data_(rows * cols) {}

  Field(uword rows, uword cols, uword slices)
      : n_rows_(rows), n_cols_(cols), n_slices_(slices), data_(rows * cols * slices) {}

  [[nodiscard]] uword n_rows() const { return n_rows_; }
  [[nodiscard]] uword n_cols() const { return n_cols_; }
  [[nodiscard]] uword n_slices() const { return n_slices_; }
  [[nodiscard]] uword n_elem() const { return data_.size(); }
  [[nodiscard]] bool is_empty() const { return data_.empty(); }

  void resize(uword rows, uword cols, uword slices = 1) {
    n_rows_ = rows;
    n_cols_ = cols;
    n_slices_ = slices;
    data_.resize(rows * cols * slices);
  }

  T& operator()(uword idx) {
    DINGO_ASSERT(idx < data_.size(), "dingo::Field linear index out of range");
    return data_[idx];
  }

  const T& operator()(uword idx) const {
    DINGO_ASSERT(idx < data_.size(), "dingo::Field linear index out of range");
    return data_[idx];
  }

  T& operator()(uword row, uword col, uword slice = 0) {
    DINGO_ASSERT(row < n_rows_ && col < n_cols_ && slice < n_slices_, "dingo::Field index out of range");
    return data_[linear_index(row, col, slice)];
  }

  const T& operator()(uword row, uword col, uword slice = 0) const {
    DINGO_ASSERT(row < n_rows_ && col < n_cols_ && slice < n_slices_, "dingo::Field index out of range");
    return data_[linear_index(row, col, slice)];
  }

  void fill(const T& value) {
    for (auto& x : data_) {
      x = value;
    }
  }

 private:
  [[nodiscard]] uword linear_index(uword row, uword col, uword slice) const {
    return row + n_rows_ * (col + n_cols_ * slice);
  }

  uword n_rows_{0};
  uword n_cols_{0};
  uword n_slices_{0};
  std::vector<T> data_;
};

template <typename T>
using field = Field<T>;

}  // namespace dingo
