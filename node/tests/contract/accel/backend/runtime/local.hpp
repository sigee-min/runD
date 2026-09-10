#pragma once

#include "../run/local.hpp"

#include <accel/device.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace node_accel_contract::backend_runtime {

inline constexpr std::uint64_t kCounterMaximum =
    std::numeric_limits<std::uint64_t>::max();

void OverflowCounter(std::uint64_t &counter) noexcept;
[[nodiscard]] bool
CountersEqual(std::uint64_t expected,
              std::initializer_list<std::uint64_t> values) noexcept;
[[nodiscard]] rund::AccelDevice Pick(rund::AccelApi api);

[[nodiscard]] bool CheckPickTokenAdmission();
[[nodiscard]] bool CheckVulkanCommandFailure();
[[nodiscard]] bool CheckCpuCounters(const rund::AccelDevice &pick);
[[nodiscard]] bool CheckMetalCounters(const rund::AccelDevice &pick);
[[nodiscard]] bool CheckVulkanCounters(const rund::AccelDevice &pick);
[[nodiscard]] bool CheckMetalHostReadback();
[[nodiscard]] bool CheckVulkanHostReadback();
[[nodiscard]] bool CheckVulkanMemoryTier(const rund::AccelDevice &pick);
[[nodiscard]] bool CheckMetalRuntime(const rund::AccelDevice &pick);
[[nodiscard]] bool CheckVulkanRuntime();

} // namespace node_accel_contract::backend_runtime
