#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>

namespace node_accel_contract::backend {

[[nodiscard]] inline rund::AccelPolicy
Policy(const std::initializer_list<rund::AccelApi> preferred,
       const bool allow_fake = false) {
  rund::AccelPolicy policy{};
  for (rund::AccelApi &api : policy.preferred) {
    api = rund::AccelApi::Auto;
  }
  std::size_t index = 0u;
  for (const rund::AccelApi api : preferred) {
    if (index < std::size(policy.preferred)) {
      policy.preferred[index] = api;
    }
    ++index;
  }
  policy.preferred_count = static_cast<std::uint32_t>(index);
  policy.allow_fake = allow_fake;
  return policy;
}

} // namespace node_accel_contract::backend
