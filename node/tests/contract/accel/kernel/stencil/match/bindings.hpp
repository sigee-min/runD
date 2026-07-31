#pragma once

#include <accel/kernel/run/binding.hpp>

#include "resources.hpp"

namespace node_accel_contract::stencil::match {

template <typename T, std::size_t Count>
[[nodiscard]] std::array<rund::AccelRunBinding, 2u>
Bindings(const Resources<T, Count> &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.input,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.output,
              .role = rund::kernel::BufferRole::Write,
          }};
}

} // namespace node_accel_contract::stencil::match
