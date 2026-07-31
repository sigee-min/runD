#pragma once

#include <kernel/program/compute/compact/plan.hpp>

#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/value.hpp>

#include "../primitive/local.hpp"

#include <array>

namespace node_accel_contract {

[[nodiscard]] constexpr std::array<rund::kernel::u32, 8u>
CompactFixtureA() noexcept {
  return std::array<rund::kernel::u32, 8u>{1u, 0u, 5u, 0u, 1u, 9u, 0u, 1u};
}

[[nodiscard]] constexpr std::array<rund::kernel::u32, 8u>
CompactFixtureB() noexcept {
  return std::array<rund::kernel::u32, 8u>{0u, 2u, 0u, 8u, 0u, 8u, 0u, 1u};
}

[[nodiscard]] bool CompactMatchesCpuReference(
    const rund::AccelDevice &pick, rund::kernel::ComputeScalar scalar,
    const std::array<rund::kernel::u32, 8u> &flags, std::uint64_t capacity);

[[nodiscard]] bool
CompactRejectsCapacityInsufficient(const rund::AccelDevice &pick,
                                   rund::kernel::ComputeScalar scalar);

[[nodiscard]] bool CompactBackendsAgree(
    const rund::AccelDevice &metal, const rund::AccelDevice &vulkan,
    rund::kernel::ComputeScalar scalar,
    const std::array<rund::kernel::u32, 8u> &flags, std::uint64_t capacity);

} // namespace node_accel_contract

namespace node_accel_contract::compact {

struct RunContext {
  rund::AccelContext context{};
  rund::AccelBuffer flags{};
  rund::AccelBuffer output{};
  rund::kernel::CompactPlan plan{};
  rund::AccelKernel kernel{};
  bool valid = false;
};

[[nodiscard]] RunContext Compile(const rund::AccelDevice &pick,
                                 rund::kernel::ComputeScalar scalar,
                                 const std::array<rund::kernel::u32, 8u> &flags,
                                 std::uint64_t capacity);

[[nodiscard]] rund::AccelEvidence Run(const RunContext &ctx,
                                      std::uint64_t tile_count);

} // namespace node_accel_contract::compact
