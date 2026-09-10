#pragma once

#include "../local.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/residency/execution/receipt.hpp"
#include "src/compute/device/residency/execution/run.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/device/residency/registry/execution_owner.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace rund_node_test_pipeline_residency::execution_test {

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;
using rund::compute::Reason;
using rund::compute::Status;

[[nodiscard]] residency::CacheKey InputKey();
[[nodiscard]] residency::CacheKey OutputKey();

[[nodiscard]] residency::FrameRegion region(residency::FrameTier tier,
                                            residency::FrameRole role,
                                            std::uint32_t first);

[[nodiscard]] execution::Request MakeRequest();
[[nodiscard]] execution::Request MakeFootprintRequest();
[[nodiscard]] execution::Request MakeOneRequest(
    const execution::Request &request);

[[nodiscard]] std::array<
    std::pair<residency::FrameTier, residency::FrameRole>, 8u>
FrameOwners();

[[nodiscard]] bool dependency(const execution::Plan &plan,
                              execution::NodeId node,
                              execution::NodeId expected);

[[nodiscard]] bool RegisterOwners(residency::Authority &authority,
                                  std::uint32_t count = 3u);
[[nodiscard]] bool SeedCachedInput(residency::Authority &authority,
                                   residency::CacheKey key);

[[nodiscard]] int CheckExecutionPlan();
[[nodiscard]] int CheckExecutionAuthority();
[[nodiscard]] int CheckExecutionNative();

} // namespace rund_node_test_pipeline_residency::execution_test
