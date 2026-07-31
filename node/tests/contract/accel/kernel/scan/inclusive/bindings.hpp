#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::scan_inclusive {

template <typename T>
[[nodiscard]] std::array<rund::AccelRunBinding, 2u>
Bindings(const Resources<T> &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.read,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.write,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::scan_inclusive
