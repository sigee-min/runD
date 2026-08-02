#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "kernel/bindings/histogram.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/histogram/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct MetalKernelImmutablePipelines;
struct VulkanAdapter;
struct VulkanKernelImmutablePipelines;

[[nodiscard]] rund::AccelCheck
ExecuteMetalHistogram(const rund::AccelDevice &pick,
                      const rund::kernel::HistogramDesc &desc,
                      const rund::kernel::HistogramPlan &plan,
                      const HistogramBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareMetalHistogram(const rund::AccelDevice &pick,
                      const rund::kernel::HistogramDesc &desc,
                      const rund::kernel::HistogramPlan &plan,
                      const HistogramBinds &bindings,
                      std::shared_ptr<void> &resources,
                      const MetalKernelImmutablePipelines *pipelines =
                          nullptr);
[[nodiscard]] rund::AccelCheck EncodeMetalHistogram(MetalAdapter &adapter,
                                         const std::shared_ptr<void> &resources,
                                         void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalHistogram(MetalAdapter &adapter,
                     const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanHistogram(const rund::AccelDevice &pick,
                       const rund::kernel::HistogramDesc &desc,
                       const rund::kernel::HistogramPlan &plan,
                       const HistogramBinds &bindings);
[[nodiscard]] rund::AccelCheck
PrepareVulkanHistogram(const rund::AccelDevice &pick,
                       const rund::kernel::HistogramDesc &desc,
                       const rund::kernel::HistogramPlan &plan,
                       const HistogramBinds &bindings,
                       std::shared_ptr<void> &resources,
                       const VulkanKernelImmutablePipelines *pipelines =
                           nullptr);
[[nodiscard]] rund::AccelCheck
EncodeVulkanHistogram(VulkanAdapter &adapter,
                      const std::shared_ptr<void> &resources,
                      void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanHistogram(VulkanAdapter &adapter,
                      const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
