#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::gather::reject {

[[nodiscard]] inline std::array<rund::AccelRunBinding, 3u>
Bindings(const Resources &resources) {
  return {
      rund::AccelRunBinding{
          .buffer = &resources.source,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.index,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
}

[[nodiscard]] inline std::array<rund::AccelRunBinding, 4u>
BoundedBindings(const Resources &resources) {
  return {
      rund::AccelRunBinding{
          .buffer = &resources.source,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.index,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.count,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &resources.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
}

} // namespace node_accel_contract::gather::reject
