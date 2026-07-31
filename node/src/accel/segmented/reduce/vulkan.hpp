#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/bindings/segmented.hpp"
#include <kernel/program/compute/segmented/reduce/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;

[[nodiscard]] rund::AccelCheck
ExecuteVulkanSegmentedReduce(const rund::AccelDevice &pick,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             rund::kernel::ComputeDomain domain,
                             const SegmentedReduceBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareVulkanSegmentedReduce(const rund::AccelDevice &pick,
                             const rund::kernel::SegmentedReduceDesc &desc,
                             const rund::kernel::SegmentedReducePlan &plan,
                             rund::kernel::ComputeDomain domain,
                             const SegmentedReduceBinds &bindings,
                             std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck
EncodeVulkanSegmentedReduce(VulkanAdapter &adapter,
                            const std::shared_ptr<void> &resources,
                            void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanSegmentedReduce(VulkanAdapter &adapter,
                            const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
