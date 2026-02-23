#pragma once

#if defined(__has_include)
#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#elif __has_include("../../../third_party/eigen/Eigen/Dense")
#include "../../../third_party/eigen/Eigen/Dense"
#else
#error "Eigen/Dense not found. Install Eigen or vendor third_party/eigen."
#endif
#else
#include <Eigen/Dense>
#endif

