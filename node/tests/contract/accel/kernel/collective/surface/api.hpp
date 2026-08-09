#pragma once

namespace node_accel_contract {

[[nodiscard]] bool AccelGraphKernelCollectiveContract();
[[nodiscard]] bool RequiredMetalRunsInclusiveScan();
[[nodiscard]] bool RequiredVulkanRunsInclusiveScan();
[[nodiscard]] bool RequiredMetalRunsBoundedSort();
[[nodiscard]] bool RequiredVulkanRunsBoundedSort();
[[nodiscard]] bool RequiredMetalRunsGather();
[[nodiscard]] bool RequiredVulkanRunsGather();
[[nodiscard]] bool RequiredMetalRunsHistogram();
[[nodiscard]] bool RequiredVulkanRunsHistogram();
[[nodiscard]] bool AccelGraphKernelPartitionCompileContract();
[[nodiscard]] bool RequiredMetalRunsPartition();
[[nodiscard]] bool RequiredVulkanRunsPartition();
[[nodiscard]] bool RequiredMetalRunsReduce();
[[nodiscard]] bool RequiredVulkanRunsReduce();
[[nodiscard]] bool RequiredMetalRunsScatter();
[[nodiscard]] bool RequiredVulkanRunsScatter();
[[nodiscard]] bool RequiredMetalRunsStencil();
[[nodiscard]] bool RequiredVulkanRunsStencil();
[[nodiscard]] bool WindowProjectionContract();
[[nodiscard]] bool RequiredMetalRunsWindow();
[[nodiscard]] bool RequiredVulkanRunsWindow();

} // namespace node_accel_contract
