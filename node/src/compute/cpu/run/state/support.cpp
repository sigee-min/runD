#include "local.hpp"

#include <limits>

namespace rund::compute::detail::cpu_run_state_detail {

bool add_count(std::size_t &total, const std::size_t value) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += value;
  return true;
}

} // namespace rund::compute::detail::cpu_run_state_detail
