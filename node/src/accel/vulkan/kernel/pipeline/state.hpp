#pragma once

#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/submission.hpp"
#include "../../collective/pipeline.hpp"
#include "../../command/model.hpp"
#include "../control.hpp"
#include "../local.hpp"
#include "../publish.hpp"
#include "../window.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineTelemetryRecord final {
  VulkanPipelineTelemetrySource source{};
  std::shared_ptr<void> owner;
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  std::uint32_t declared_step{};
};

struct VulkanPipelineTelemetryParams final {
  std::uint32_t kind{};
  std::uint32_t primary_word_count{};
  std::uint32_t count_source{};
  std::uint32_t predicate_source{};
  std::uint32_t has_count{};
  std::uint32_t has_predicate{};
  std::uint32_t iteration{};
  std::uint32_t count_word_offset{};
  std::uint32_t predicate_word_offset{};
  std::uint32_t indirect_dispatch_count{};
  std::uint32_t declared_step{};
  std::uint64_t capacity{};
  std::uint64_t predicate_expected{};
  std::uint64_t work_item_count{};
};

static_assert(sizeof(VulkanPipelineTelemetryParams) == 72u);

struct VulkanPipelineProfileTelemetryParams final {
  VulkanPipelineTelemetryParams telemetry{};
  std::uint32_t declared_step_count{};
};

static_assert(sizeof(VulkanPipelineProfileTelemetryParams) == 80u);

struct VulkanPipelineProfile final {
  VkQueryPool timestamps{VK_NULL_HANDLE};
  std::array<PreparedPipelineStepEvidence, PreparedPipelineStepCapacity> rows{};
  std::array<PreparedPipelineStepControl, PreparedPipelineStepCapacity>
      controls{};
  std::array<bool, PreparedPipelineStepCapacity> timestamped{};
  std::array<std::uint32_t, PreparedPipelineStepCapacity> declared_steps{};
  std::uint32_t active_step_count{};
  std::uint32_t declared_step_count{};
  std::uint32_t query_count{};
  std::uint64_t instrumentation_command_count{};
  std::uint64_t instrumentation_byte_count{};
};

struct VulkanPipeline final {
  VulkanAdapter *adapter{};
  VulkanCommand command{};
  VulkanPipelineControlResources control{};
  VulkanPipelinePublishResources publish{};
  VulkanWindowResources window{};
  std::unique_ptr<VulkanPipelineProfile> profile;
  VulkanCollectivePipeline *telemetry_pipeline{};
  std::vector<VulkanPipelineTelemetryRecord> telemetry;
  std::shared_ptr<void> recurrence;
  std::uint64_t dispatch_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  submission::State<VulkanPipeline> submission{};

  ~VulkanPipeline();
};

[[nodiscard]] rund::AccelCheck
FailVulkanPipeline(std::shared_ptr<VulkanPipeline> &pipeline,
                   const char *reason);
[[nodiscard]] bool ValidVulkanPipeline(const VulkanPipeline *pipeline) noexcept;

#endif

} // namespace rund::node::accel::detail
