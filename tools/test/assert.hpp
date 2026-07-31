#pragma once

namespace rund::test {

[[noreturn]] void fail(const char *condition, const char *file,
                       int line) noexcept;

} // namespace rund::test

#define TEST_ASSERT(condition)                                                 \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ::rund::test::fail(#condition, __FILE__, __LINE__);                       \
    }                                                                          \
  } while (false)
