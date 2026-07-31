#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::scatter::match {

template <typename T>
[[nodiscard]] std::array<rund::AccelRunBinding, 3u>
Bindings(const Resources<T> &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.values,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.index,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.output,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::scatter::match
