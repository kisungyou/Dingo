#pragma once

#include <cstddef>
#include <limits>

namespace dingo {

using uword = std::size_t;

struct all_t {
  explicit constexpr all_t() = default;
};

inline constexpr all_t all_idx{};

struct span {
  uword begin{0};
  uword end{0};
};

inline constexpr span range(uword begin, uword end) { return span{begin, end}; }

inline constexpr uword span_all_end() { return std::numeric_limits<uword>::max(); }

}  // namespace dingo
