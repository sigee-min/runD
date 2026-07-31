#include "test/assert.hpp"

#include <cstdio>
#include <cstdlib>

namespace rund::test {

[[noreturn]] void fail(const char *const condition, const char *const file,
                       const int line) noexcept {
  std::fprintf(stderr, "TEST_ASSERT failed: %s at %s:%d\n", condition, file,
               line);
  std::exit(EXIT_FAILURE);
}

} // namespace rund::test
