#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/partition.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/partition/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;
struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck
ExecuteMetalPartition(const rund::AccelDevice &pick,
                      const rund::kernel::PartitionDesc &desc,
                      const rund::kernel::PartitionPlan &plan,
                      const PartitionBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareMetalPartition(const rund::AccelDevice &pick,
                      const rund::kernel::PartitionDesc &desc,
                      const rund::kernel::PartitionPlan &plan,
                      const PartitionBinds &bindings,
                      std::shared_ptr<void> &resources,
                      const MetalKernelImmutablePipelines *pipelines =
                          nullptr);
[[nodiscard]] rund::AccelCheck EncodeMetalPartition(MetalAdapter &adapter,
                                         const std::shared_ptr<void> &resources,
                                         void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalPartition(MetalAdapter &adapter,
                     const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanPartition(const rund::AccelDevice &pick,
                       const rund::kernel::PartitionDesc &desc,
                       const rund::kernel::PartitionPlan &plan,
                       const PartitionBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareVulkanPartition(const rund::AccelDevice &pick,
                       const rund::kernel::PartitionDesc &desc,
                       const rund::kernel::PartitionPlan &plan,
                       const PartitionBinds &bindings,
                       std::shared_ptr<void> &resources,
                       const VulkanKernelImmutablePipelines *pipelines =
                           nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanPartition(VulkanAdapter &adapter,
                      const std::shared_ptr<void> &resources,
                      void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanPartition(VulkanAdapter &adapter,
                      const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
