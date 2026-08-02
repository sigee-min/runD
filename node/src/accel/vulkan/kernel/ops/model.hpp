#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../kernel/preparation.hpp"
#include "../../adapter/api.hpp"
#include "../../kernel.hpp"

#include <array>
#include <cstdint>
#include <memory>

#include <kernel/program/compute/graph/schema.hpp>

namespace rund::node::accel::detail {

struct VulkanBuffer;
struct VulkanPipelineTelemetrySource;
struct VulkanKernelImmutablePipelines;

enum class VulkanPipelineStatusRule : std::uint32_t {
  None,
  Exact,
  BitFlags,
  LowBit,
  AnyFailure,
};

struct VulkanPipelineStatusSource final {
  const VulkanBuffer *raw{};
  std::uint64_t offset{};
  std::uint32_t count{};
  VulkanPipelineStatusRule rule{VulkanPipelineStatusRule::None};
  std::uint32_t success{};
  std::array<std::uint32_t, 8u> raw_values{};
  std::array<std::uint32_t, 8u> reasons{};
  std::uint32_t mapping_count{};
};

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

using VulkanPrepareStepFn = rund::AccelCheck (*)(
    const rund::AccelDevice &pick, const BoundStep &step,
    KernelPreparationMode mode, const VulkanKernelImmutablePipelines *pipelines,
    std::shared_ptr<void> &resources);
using VulkanEncodeStepFn = rund::AccelCheck (*)(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    void *command_buffer);
using VulkanFinishStepFn = rund::AccelCheck (*)(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources);
using VulkanPipelineStatusFn = rund::AccelCheck (*)(
    const std::shared_ptr<void> &resources, VulkanPipelineStatusSource &source);

enum class VulkanPipelineTelemetryKind : std::uint8_t {
  None,
  ControlledMap,
  IndexedControl,
  ControlledCollective,
  GatherControl,
};

struct VulkanPipelineTelemetrySource final {
  VulkanPipelineTelemetryKind kind{VulkanPipelineTelemetryKind::None};
  const VulkanBuffer *primary{};
  const VulkanBuffer *count{};
  const VulkanBuffer *predicate{};
  rund::kernel::GraphControl control{};
  std::uint64_t count_offset{};
  std::uint64_t predicate_offset{};
  std::uint64_t capacity{};
  std::uint64_t work_item_count{};
  std::uint32_t primary_word_count{};
  std::uint32_t indirect_dispatch_count{};
};

using VulkanPipelineTelemetryFn =
    rund::AccelCheck (*)(const std::shared_ptr<void> &resources,
                         VulkanPipelineTelemetrySource &source);
using VulkanFailureFn = bool (*)(const std::shared_ptr<void> &resources,
                                 std::uint64_t &ordinal) noexcept;
// Exact number of device-authored indirect dispatches emitted by one prepared
// step. Pipeline window capture consumes one gate descriptor per such command,
// so this encoder-adjacent contract is the sole preallocation authority.
using VulkanPipelineCaptureDemandFn =
    rund::AccelCheck (*)(const std::shared_ptr<void> &resources,
                         std::uint64_t &indirect_dispatch_count) noexcept;

struct VulkanKernelOps {
  VulkanPrepareStepFn prepare = nullptr;
  VulkanEncodeStepFn encode = nullptr;
  VulkanFinishStepFn finish = nullptr;
  VulkanPipelineStatusFn pipeline_status = nullptr;
  VulkanPipelineTelemetryFn pipeline_telemetry = nullptr;
  VulkanFailureFn failure = nullptr;
  VulkanPipelineCaptureDemandFn pipeline_capture_demand = nullptr;
};

#endif

} // namespace rund::node::accel::detail
