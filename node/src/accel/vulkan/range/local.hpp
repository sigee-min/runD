#pragma once

#include "../../range_aggregate/execution/projection.hpp"
#include "../adapter/state.hpp"
#include "../barrier.hpp"
#include "../buffer/resident/model.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include "api.hpp"
#include <accel/check.hpp>
#include <kernel/program/compute/graph/schema.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanKernelImmutablePipelines;
struct BoundControl;

struct VulkanRangeControlPush final {
  std::uint32_t count_word{};
  std::uint32_t param_stride_words{};
};

static_assert(sizeof(VulkanRangeControlPush) == 2u * sizeof(std::uint32_t));

// Primitive adapters own public binding semantics.  This value carries only
// already authenticated generic storage bindings into RangeExec.
class VulkanRangeBinds final {
public:
  VulkanRangeBinds() = delete;

  [[nodiscard]] static std::optional<VulkanRangeBinds>
  make(const VulkanResidentBufferResult &input,
       const VulkanResidentBufferResult &output) noexcept {
    if (!input.check.ok || !output.check.ok || input.device_buffer == nullptr ||
        output.device_buffer == nullptr) {
      return std::nullopt;
    }
    const VulkanStorageBinding input_binding =
        VulkanStorageBindingFor(input.device_buffer, input.ref);
    const VulkanStorageBinding output_binding =
        VulkanStorageBindingFor(output.device_buffer, output.ref);
    if (input_binding.buffer == nullptr || output_binding.buffer == nullptr) {
      return std::nullopt;
    }
    return VulkanRangeBinds{input.device_buffer, output.device_buffer,
                            input_binding, output_binding};
  }

  [[nodiscard]] const VulkanBuffer *input() const noexcept { return input_; }
  [[nodiscard]] const VulkanBuffer *output() const noexcept { return output_; }
  [[nodiscard]] const VulkanStorageBinding &input_binding() const noexcept {
    return input_binding_;
  }
  [[nodiscard]] const VulkanStorageBinding &output_binding() const noexcept {
    return output_binding_;
  }

private:
  VulkanRangeBinds(const VulkanBuffer *input, const VulkanBuffer *output,
                   const VulkanStorageBinding input_binding,
                   const VulkanStorageBinding output_binding) noexcept
      : input_(input), output_(output), input_binding_(input_binding),
        output_binding_(output_binding) {}

  const VulkanBuffer *input_;
  const VulkanBuffer *output_;
  VulkanStorageBinding input_binding_;
  VulkanStorageBinding output_binding_;
};

struct VulkanRangeResources {
  VulkanAdapter *adapter = nullptr;
  RangePlan range{RangePlan::rejected("compute_range_aggregate_unavailable")};
  std::array<VulkanCollectivePipeline *, kRangeStageCap> pipelines{};
  std::array<VulkanBuffer, kRangeStageCap> params{};
  std::array<VkDescriptorSet, kRangeStageCap> descriptor_sets{};
  std::array<VulkanBuffer, kRangeTempCap> temporaries{};
  VulkanResidentBufferResult control_count{};
  VulkanBuffer control_params{};
  VulkanBuffer control_indirect{};
  VulkanStatus control_status{};
  VulkanCollectivePipeline *control_pipeline = nullptr;
  VkDescriptorSet control_descriptor = VK_NULL_HANDLE;
  rund::kernel::GraphControl control{};
  VkDeviceSize control_param_stride{};
  VulkanRangeControlPush control_push{};
  std::uint32_t stage_count{};
  const VulkanBuffer *input = nullptr;
  const VulkanBuffer *output = nullptr;
  VulkanStorageBinding input_binding{};
  VulkanStorageBinding output_binding{};
  bool controlled{};
};

void DestroyVulkanRangeResources(void *raw);
[[nodiscard]] std::string VulkanRangeSource(const RangeExec &execution);
[[nodiscard]] bool VulkanRangeSourceBytes(const RangeExec &execution,
                                          std::uint64_t &bytes) noexcept;
[[nodiscard]] bool VulkanRangeSourceMatches(const RangeExec &execution,
                                            std::string_view source,
                                            std::uint64_t source_hash) noexcept;
[[nodiscard]] std::string VulkanRangeControlSource(const RangePlan &plan);
[[nodiscard]] bool VulkanRangeControlSourceBytes(const RangePlan &plan,
                                                 std::uint64_t &bytes) noexcept;
[[nodiscard]] bool
VulkanRangeControlSourceMatches(const RangePlan &plan, std::string_view source,
                                std::uint64_t source_hash) noexcept;
[[nodiscard]] bool
VulkanRangePipelineMatches(const VulkanAdapter &adapter,
                           const VulkanCollectivePipeline *pipeline,
                           const RangeExec &execution) noexcept;
[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanRangePipeline(VulkanAdapter &adapter, const RangeExec &execution);
[[nodiscard]] VulkanCollectivePipeline *
AcquireVulkanRangeControlPipeline(VulkanAdapter &adapter,
                                  const RangePlan &plan);
[[nodiscard]] bool
VulkanRangeControlPipelineMatches(const VulkanAdapter &adapter,
                                  const VulkanCollectivePipeline *pipeline,
                                  const RangePlan &plan) noexcept;
[[nodiscard]] bool CreateVulkanRangeDescriptors(VulkanAdapter &adapter,
                                                VulkanRangeResources &resources,
                                                std::uint32_t stage_index);
[[nodiscard]] rund::AccelCheck
EncodeVulkanRange(VulkanAdapter &adapter,
                  const std::shared_ptr<void> &resources, void *command_buffer);
[[nodiscard]] rund::AccelCheck
FinishVulkanRange(VulkanAdapter &adapter,
                  const std::shared_ptr<void> &resources);
[[nodiscard]] rund::AccelCheck PrepareVulkanRange(
    const rund::AccelDevice &pick, const RangePlan &range,
    const VulkanRangeBinds &bindings, rund::kernel::NodeKind owner_kind,
    std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *pipelines = nullptr,
    const BoundControl *control = nullptr,
    KernelPreparationMode mode = KernelPreparationMode::Standalone);
[[nodiscard]] bool VulkanRangeScratch(const VulkanRangeResources &resources,
                                      std::uint32_t stage_index,
                                      const VulkanBuffer *&scratch0,
                                      const VulkanBuffer *&scratch1) noexcept;

#endif

} // namespace rund::node::accel::detail
