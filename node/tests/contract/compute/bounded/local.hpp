#pragma once

#include <rund/compute.hpp>

#include <cstdint>

namespace rund_node_bounded_contract {

[[nodiscard]] int CheckFilterBackend(rund::compute::Target,
                                     rund::compute::Stats * = nullptr);
[[nodiscard]] int CheckAcceleratorFilterLaws(rund::compute::Backend);
[[nodiscard]] int CheckTypedBoundedMap(rund::compute::Target);
[[nodiscard]] int CheckInactiveTail(rund::compute::Target);
[[nodiscard]] int CheckReduceRewrite(rund::compute::Target);
[[nodiscard]] int CheckExpandBackend(rund::compute::Backend,
                                     rund::compute::Stats *);
[[nodiscard]] int CheckGroupRewriteBackend(rund::compute::Backend,
                                           std::uint64_t &, std::uint64_t &,
                                           std::uint64_t &);
[[nodiscard]] int CheckCompactCapacityBackend(rund::compute::Backend);
[[nodiscard]] int CheckVulkanPartitionPipelineWidths();
[[nodiscard]] int CheckInvalidBoundedCount(rund::compute::Backend);
[[nodiscard]] bool CheckInputSetMapShapeAdmission();
[[nodiscard]] int CheckCpu();
[[nodiscard]] int CheckParity();
[[nodiscard]] int CheckBackends();

} // namespace rund_node_bounded_contract
