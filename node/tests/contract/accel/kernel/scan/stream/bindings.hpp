#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

#include <array>

namespace node_accel_contract::scan_stream {

[[nodiscard]] inline std::array<rund::AccelRunBinding, 4u>
BuildBindings(Resources &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.read,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.mid,
              .role = rund::kernel::BufferRole::Write,
          },
          rund::AccelRunBinding{
              .buffer = &resources.mid,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.write,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::scan_stream
