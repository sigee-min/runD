#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::collective::sort_reject {

template <typename Key>
[[nodiscard]] std::array<rund::AccelRunBinding, 4u>
Bindings(Resources<Key> &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.run_read_keys,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.run_read_values,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.run_write_keys,
              .role = rund::kernel::BufferRole::Write,
          },
          rund::AccelRunBinding{
              .buffer = &resources.run_write_values,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::collective::sort_reject
