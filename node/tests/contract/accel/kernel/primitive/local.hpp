#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>

#include "reason.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace node_accel_contract::primitive {

[[nodiscard]] constexpr rund::AccelPolicy
Policy(const rund::AccelApi api) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

[[nodiscard]] constexpr rund::AccelBufferDesc
BufferDesc(const rund::BufferUsage usage,
           const std::uint64_t scalar_width_bytes,
           const std::uint64_t count) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = scalar_width_bytes,
      .count = count,
      .usage = usage,
  };
}

template <typename T>
[[nodiscard]] std::uint64_t HashValues(const T *const values,
                                       const std::size_t count) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  const auto *const bytes = reinterpret_cast<const std::uint8_t *>(values);
  for (std::size_t index = 0u; index < count * sizeof(T); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

[[nodiscard]] inline bool
EvidenceReason(const rund::AccelEvidence &evidence,
               const std::string_view reason) noexcept {
  return !evidence.ok && std::string_view{evidence.reason} == reason;
}

} // namespace node_accel_contract::primitive
