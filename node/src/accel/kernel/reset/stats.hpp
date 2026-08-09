#pragma once

#include <accel/runtime.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

inline void SetResetStats(rund::RuntimeStats &stats, const bool ok,
                          const std::uint64_t commands,
                          const std::uint64_t bytes) noexcept {
  stats.run.work.reset_command_count = ok ? commands : 0u;
  stats.run.work.reset_bytes = ok ? bytes : 0u;
}

} // namespace rund::node::accel::detail
