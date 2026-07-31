#pragma once

#include <accel/check.hpp>

#include <cstddef>
#include <limits>

namespace rund::node::accel::detail::finish {

template <class Resources, class Step>
[[nodiscard]] rund::AccelCheck Steps(Resources &resources, Step &&step,
                                     std::size_t *const failed = nullptr) {
  if (failed != nullptr) {
    *failed = std::numeric_limits<std::size_t>::max();
  }
  rund::AccelCheck result{true, "ok"};
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    auto *const entry = resources.entry(index);
    if (entry == nullptr) {
      if (failed != nullptr) {
        *failed = index;
      }
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const rund::AccelCheck current = step(*entry);
    if (!current.ok) {
      if (failed != nullptr) {
        *failed = index;
      }
      return current;
    }
    if (current.failed_batches != 0u) {
      if (result.failed_batches == 0u) {
        result.first_failed_batch = current.first_failed_batch;
        result.first_status = current.first_status;
      }
      result.failed_batches += current.failed_batches;
    }
  }
  return result;
}

} // namespace rund::node::accel::detail::finish
