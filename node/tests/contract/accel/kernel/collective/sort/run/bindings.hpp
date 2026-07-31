#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::collective::sort_run {

template <typename Key, std::size_t Count>
[[nodiscard]] std::array<rund::AccelRunBinding, 4u>
Bindings(Resources<Key, Count> &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.read_keys,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.read_values,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.write_keys,
              .role = rund::kernel::BufferRole::Write,
          },
          rund::AccelRunBinding{
              .buffer = &resources.write_values,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::collective::sort_run
