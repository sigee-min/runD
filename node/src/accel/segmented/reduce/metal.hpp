#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/bindings/segmented.hpp"
#include "../../kernel/memory.hpp"
#include <kernel/program/compute/segmented/reduce/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalPipelineStatusBindings;
struct MetalKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck
ExecuteMetalSegmentedReduce(const rund::AccelDevice &pick,
                            const rund::kernel::SegmentedReduceDesc &desc,
                            const rund::kernel::SegmentedReducePlan &plan,
                            rund::kernel::ComputeDomain domain,
                            const SegmentedReduceBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareMetalSegmentedReduce(const rund::AccelDevice &pick,
                            const rund::kernel::SegmentedReduceDesc &desc,
                            const rund::kernel::SegmentedReducePlan &plan,
                            rund::kernel::ComputeDomain domain,
                            const SegmentedReduceBinds &bindings,
                            std::shared_ptr<void> &resources,
                            const MetalKernelImmutablePipelines *pipelines =
                                nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalSegmentedReduce(MetalAdapter &adapter,
                           const std::shared_ptr<void> &resources,
                           void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalSegmentedReduce(MetalAdapter &adapter,
                           const std::shared_ptr<void> &resources);
[[nodiscard]] PreparedMemory
MetalSegmentedReduceMemory(const std::shared_ptr<void> &resources,
                           std::uint64_t budget) noexcept;
[[nodiscard]] bool DescribeMetalSegmentedReducePipelineStatus(
    const std::shared_ptr<void> &resources,
    MetalPipelineStatusBindings &bindings) noexcept;

} // namespace rund::node::accel::detail
