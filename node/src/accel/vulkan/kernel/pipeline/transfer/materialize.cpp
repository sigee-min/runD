#include "../../../adapter/error.hpp"
#include "../../../buffer/access.hpp"
#include "../../../buffer/create.hpp"
#include "../../../../backend/result.hpp"

#include "../transfer.hpp"

#include "../../../buffer/create/telemetry.hpp"
#include "../../../buffer/resident/find.hpp"
#include "../../../buffer/transfer/range.hpp"
#include "../../../command/resources.hpp"
#include "../../../resident/access.hpp"
#include "../prepare/record.hpp"
#include "../residency/local.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
QueryVulkanPipelineResidency(const std::shared_ptr<void> &prepared,
                             bool &supported) noexcept {
  supported = false;
  const auto *const pipeline =
      static_cast<const VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline)) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  supported = pipeline->residency != nullptr && pipeline->residency->ready;
  return rund::AccelCheck{true, "ok"};
}
namespace {

struct VulkanPipelineTransferCandidate final {
  VulkanPipeline *pipeline{};
  VulkanCommand command{};
  VulkanPipelineTransfer transfer{};
  std::uint64_t retained_bytes{};
  bool staging_reused{};

  ~VulkanPipelineTransferCandidate() {
    if (pipeline == nullptr || pipeline->adapter == nullptr) {
      return;
    }
    DestroyCommand(pipeline->adapter->device, transfer.compute);
    DestroyCommand(pipeline->adapter->device, command);
    ReleaseVulkanMemoryLease(*pipeline->adapter, transfer.staging);
    DestroyVulkanBuffer(*pipeline->adapter, transfer.staging);
  }

  void publish() noexcept {
    VulkanPipeline &target = *pipeline;
    DestroyCommand(target.adapter->device, target.command);
    target.command = command;
    command = {};
    target.transfer = std::move(transfer);
    transfer = {};
    target.transfer.ready = true;
    target.memory_meter->add(PreparedPipelineMemory{
        .staging =
            PreparedMemory{.current = retained_bytes,
                           .peak = retained_bytes,
                           .cumulative = retained_bytes,
                           .reused = staging_reused ? retained_bytes : 0u,
                           .budget = target.adapter->caps.staging_bytes}});
  }
};

[[nodiscard]] rund::AccelCheck
EncodeComposite(VulkanPipeline &pipeline, VulkanCommand &command,
                VulkanPipelineTransfer &transfer) noexcept {
  const rund::AccelCheck begun = BeginCommand(pipeline.adapter->device, command,
                                              CommandKind::ReusablePrimary);
  if (!begun.ok) {
    return begun;
  }
  const VkBufferMemoryBarrier source{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = transfer.staging.buffer,
      .size = transfer.input_bytes,
  };
  vkCmdPipelineBarrier(command.buffer, VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 1u,
                       &source, 0u, nullptr);
  const VkBufferCopy input_copy{.size = transfer.input_bytes};
  vkCmdCopyBuffer(command.buffer, transfer.staging.buffer, transfer.input, 1u,
                  &input_copy);
  const VkBufferMemoryBarrier input_target{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = transfer.input,
      .size = transfer.input_bytes,
  };
  vkCmdPipelineBarrier(command.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &input_target, 0u, nullptr);
  vkCmdExecuteCommands(command.buffer, 1u, &transfer.compute.buffer);
  const VkBufferMemoryBarrier output_source{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask =
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = transfer.output,
      .size = transfer.output_bytes,
  };
  vkCmdPipelineBarrier(command.buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u, nullptr, 1u,
                       &output_source, 0u, nullptr);
  const VkBufferCopy output_copy{.size = transfer.output_bytes};
  vkCmdCopyBuffer(command.buffer, transfer.output, transfer.staging.buffer, 1u,
                  &output_copy);
  const VkBufferMemoryBarrier host_target{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = transfer.staging.buffer,
      .size = transfer.output_bytes,
  };
  vkCmdPipelineBarrier(command.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0u, 0u, nullptr, 1u,
                       &host_target, 0u, nullptr);
  return EndCommand(command);
}

} // namespace

void DestroyVulkanPipelineTransfer(VulkanPipeline &pipeline) noexcept {
  if (pipeline.adapter == nullptr) {
    pipeline.transfer = {};
    return;
  }
  VulkanPipelineTransfer &transfer = pipeline.transfer;
  DestroyCommand(pipeline.adapter->device, transfer.compute);
  ReleaseVulkanBuffer(*pipeline.adapter, transfer.staging);
  transfer = {};
}

rund::AccelCheck PrepareVulkanPipelineTransfer(
    const std::shared_ptr<void> &prepared, const UploadRoute &upload,
    const DownloadRoute &download,
    const std::uint64_t exact_storage_bytes) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || upload.bytes == 0u ||
      download.bytes == 0u || upload.offset != 0u || download.offset != 0u ||
      exact_storage_bytes == 0u ||
      (upload.bytes & (kVulkanTransferAlignment - 1u)) != 0u ||
      (download.bytes & (kVulkanTransferAlignment - 1u)) != 0u) {
    return rund::AccelCheck{false, "accel_vulkan_transfer_invalid"};
  }
  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->submission.active() || pipeline->transfer.ready) {
    return rund::AccelCheck{false, "compute_pipeline_busy"};
  }
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult output{};
  {
    VulkanResidentState &resident = VulkanResidents(*pipeline->adapter);
    std::lock_guard resident_lock{resident.mutex};
    input = ResolveVulkanResidentBuffer(
        resident, upload.resident, upload.handle, "accel_buffer_unavailable");
    output = ResolveVulkanResidentBuffer(resident, download.resident,
                                         download.handle,
                                         "accel_buffer_unavailable");
  }
  if (!input.check.ok || !output.check.ok || input.device_buffer == nullptr ||
      output.device_buffer == nullptr ||
      upload.bytes > input.device_buffer->bytes ||
      download.bytes > output.device_buffer->bytes) {
    const char *const reason =
        !input.check.ok ? input.check.reason
                        : (!output.check.ok ? output.check.reason
                                            : "accel_vulkan_transfer_invalid");
    return rund::AccelCheck{false, reason};
  }

  VulkanPipelineTransferCandidate candidate{.pipeline = pipeline};
  VulkanPipelineTransfer &transfer = candidate.transfer;
  const rund::AccelCheck compute_command = CreateCommand(
      pipeline->adapter->device, pipeline->adapter->compute_queue_family,
      transfer.compute, CommandKind::ImmutableSecondary);
  if (!compute_command.ok) {
    return compute_command;
  }
  const std::uint64_t staging_bytes = std::max(upload.bytes, download.bytes);
  if (!CreateVulkanBuffer(*pipeline->adapter, staging_bytes,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          transfer.staging, &candidate.staging_reused,
                          VulkanMemoryUse::Staging, exact_storage_bytes)) {
    return rund::AccelCheck{false, VulkanLastError(pipeline->adapter)};
  }
  candidate.retained_bytes = transfer.staging.allocated_bytes;
  if (candidate.retained_bytes != exact_storage_bytes) {
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  transfer.input = input.device_buffer->buffer;
  transfer.output = output.device_buffer->buffer;
  transfer.input_storage = std::move(input.storage);
  transfer.output_storage = std::move(output.storage);
  transfer.input_bytes = upload.bytes;
  transfer.output_bytes = download.bytes;
  const rund::AccelCheck compute_recorded = RecordVulkanPipeline(
      *pipeline, transfer.compute, CommandKind::ImmutableSecondary, true);
  if (!compute_recorded.ok) {
    return compute_recorded;
  }
  const rund::AccelCheck primary_created = CreateCommand(
      pipeline->adapter->device, pipeline->adapter->compute_queue_family,
      candidate.command, CommandKind::ReusablePrimary);
  const rund::AccelCheck composite_recorded =
      primary_created.ok
          ? EncodeComposite(*pipeline, candidate.command, transfer)
          : primary_created;
  if (!composite_recorded.ok) {
    return composite_recorded;
  }
  if (pipeline->memory_meter == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  candidate.publish();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
