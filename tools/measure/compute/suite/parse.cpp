#include "core.hpp"

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
bool ParseBackend(const std::string_view name, Backend &backend) noexcept {
  for (const Backend candidate : kBackends) {
    if (name == Name(candidate)) {
      backend = candidate;
      return true;
    }
  }
  return false;
}
#endif

} // namespace rund::measure::compute
