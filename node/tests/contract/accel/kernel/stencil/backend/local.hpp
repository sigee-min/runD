#pragma once

#include "../local.hpp"
#include "src/accel/range_aggregate/plan.hpp"

#include <accel/device.hpp>

#include <cstdint>
#include <optional>

namespace node_accel_contract::stencil::backend {

enum class RuntimeSharedProbeStatus : std::uint8_t {
  Failed,
  SharedVerified,
  NoSharedCapabilityVerified,
};

struct RuntimeSharedProbe final {
  RuntimeSharedProbeStatus status{RuntimeSharedProbeStatus::Failed};
  std::optional<rund::node::accel::detail::RangeCandidate> candidate{};
};

[[nodiscard]] bool StencilMatch(bool ok, const char *name);
[[nodiscard]] std::optional<rund::node::accel::detail::RangeCandidate>
RangeExecCandidate(const rund::node::accel::detail::RangePlan &plan) noexcept;
[[nodiscard]] bool RuntimeSharedProbeMatchesContract(RuntimeSharedProbe probe,
                                                     const char *backend);

[[nodiscard]] bool RunGeometry(const rund::AccelDevice &pick);
[[nodiscard]] bool RunValue(const rund::AccelDevice &pick);
[[nodiscard]] bool RunRange(const rund::AccelDevice &pick);
[[nodiscard]] bool RunMetalRequired();
[[nodiscard]] bool RunVulkanRequired();

} // namespace node_accel_contract::stencil::backend
