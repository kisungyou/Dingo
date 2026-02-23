#pragma once

#include <stdexcept>

#ifndef DINGO_ASSERT
#if defined(DINGO_NO_DEBUG)
#define DINGO_ASSERT(condition, message) ((void)0)
#else
#define DINGO_ASSERT(condition, message)                                            \
  do {                                                                              \
    if (!(condition)) {                                                             \
      throw std::out_of_range(message);                                             \
    }                                                                               \
  } while (false)
#endif
#endif

