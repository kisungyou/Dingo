#pragma once

namespace dingo::fill {

struct zeros_t {
  explicit constexpr zeros_t() = default;
};
struct ones_t {
  explicit constexpr ones_t() = default;
};
struct randu_t {
  explicit constexpr randu_t() = default;
};
struct randn_t {
  explicit constexpr randn_t() = default;
};

inline constexpr zeros_t zeros{};
inline constexpr ones_t ones{};
inline constexpr randu_t randu{};
inline constexpr randn_t randn{};

}  // namespace dingo::fill

