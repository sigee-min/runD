#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/bindings/segmented.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;

[[nodiscard]] constexpr rund::kernel::u64 EncodedSegmentedScanDispatchCount(
    const rund::kernel::SegmentedScanPlan &plan) noexcept {
  return plan.pass_count == 2u ? 3u : (plan.pass_count == 1u ? 1u : 0u);
}

[[nodiscard]] rund::AccelCheck ExecuteMetalSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    rund::kernel::ComputeDomain domain, const SegmentedScanBinds &bindings);
[[nodiscard]] rund::AccelCheck PrepareMetalSegmentedScan(
    const rund::AccelDevice &pick, const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    rund::kernel::ComputeDomain domain, const SegmentedScanBinds &bindings,
    std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *pipelines = nullptr);
[[nodiscard]] rund::AccelCheck
EncodeMetalSegmentedScan(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources,
                         void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalSegmentedScan(MetalAdapter &adapter,
                         const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
