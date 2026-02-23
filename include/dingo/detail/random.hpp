#pragma once

#include <complex>
#include <random>
#include <type_traits>

namespace dingo::detail {

template <typename T>
inline std::mt19937_64& rng_engine() {
  static thread_local std::mt19937_64 engine{std::random_device{}()};
  return engine;
}

template <typename T>
inline T randu() {
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(0, 100);
    return dist(rng_engine<T>());
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(static_cast<T>(0), static_cast<T>(1));
    return dist(rng_engine<T>());
  } else {
    using value_t = typename T::value_type;
    return T(randu<value_t>(), randu<value_t>());
  }
}

template <typename T>
inline T randn() {
  if constexpr (std::is_integral_v<T>) {
    std::normal_distribution<double> dist(0.0, 1.0);
    return static_cast<T>(dist(rng_engine<T>()));
  } else if constexpr (std::is_floating_point_v<T>) {
    std::normal_distribution<T> dist(static_cast<T>(0), static_cast<T>(1));
    return dist(rng_engine<T>());
  } else {
    using value_t = typename T::value_type;
    return T(randn<value_t>(), randn<value_t>());
  }
}

}  // namespace dingo::detail

