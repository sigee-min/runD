#include "window.hpp"

#if !defined(RUND_NODE_HAVE_VULKAN_SDK) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL)

#else

#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/backend/usage.hpp"
#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/buffer/resident/find.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/accel/vulkan/resident/access.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] bool ProductStagedLoop(const std::size_t epochs,
                                     const ProductFault fault) {
  using namespace rund::compute;
  const std::size_t elements = epochs * 2u;
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::uint32_t>("vulkan-product-window", 1u,
                                         [](auto value) { return value + 1u; })
                     .compile();
  auto input_backing =
      std::make_shared<WindowBacking>(elements * sizeof(std::uint32_t));
  auto output_backing =
      std::make_shared<WindowBacking>(elements * sizeof(std::uint32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    std::fprintf(stderr, "vulkan product q=%zu prepare=%u\n", epochs,
                 static_cast<unsigned>(prepared.reason()));
    return prepared.reason() == Reason::BackendUnsupported;
  }
  const auto state = prepared.value();
  const std::array<std::shared_ptr<detail::PipelineState>, 2u> pipelines{
      state->pipeline, state->alternate_pipeline};
  bool coherent = true;
  std::size_t coherent_bank = 0u;
  for (const auto &pipeline : pipelines) {
    const bool structure =
        pipeline != nullptr && pipeline->device != nullptr &&
        pipeline->device->ops != nullptr &&
        pipeline->device->ops->host_read != nullptr &&
        pipeline->device->ops->host_write != nullptr &&
        pipeline->residency_input < pipeline->resources.size() &&
        pipeline->residency_output < pipeline->resources.size();
    if (!structure) {
      std::fprintf(stderr,
                   "vulkan product q=%zu bank=%zu structure=0 pipeline=%u "
                   "device=%u ops=%u resources=%zu input=%u output=%u\n",
                   epochs, coherent_bank,
                   static_cast<unsigned>(pipeline != nullptr),
                   static_cast<unsigned>(pipeline != nullptr &&
                                         pipeline->device != nullptr),
                   static_cast<unsigned>(pipeline != nullptr &&
                                         pipeline->device != nullptr &&
                                         pipeline->device->ops != nullptr),
                   pipeline == nullptr ? 0u : pipeline->resources.size(),
                   pipeline == nullptr ? 0u : pipeline->residency_input,
                   pipeline == nullptr ? 0u : pipeline->residency_output);
      coherent = false;
      break;
    }
    const auto &input_resource = pipeline->resources[pipeline->residency_input];
    const auto &output_resource =
        pipeline->resources[pipeline->residency_output];
    coherent = input_resource.buffer != nullptr &&
               output_resource.buffer != nullptr &&
               pipeline->device->ops->host_write(*pipeline->device,
                                                 *input_resource.buffer) &&
               pipeline->device->ops->host_read(*pipeline->device,
                                                *output_resource.buffer);
    if (!coherent) {
      std::fprintf(
          stderr, "vulkan product q=%zu bank=%zu buffers=%u/%u views=%u/%u\n",
          epochs, coherent_bank,
          static_cast<unsigned>(input_resource.buffer != nullptr),
          static_cast<unsigned>(output_resource.buffer != nullptr),
          static_cast<unsigned>(input_resource.buffer != nullptr &&
                                pipeline->device->ops->host_write(
                                    *pipeline->device, *input_resource.buffer)),
          static_cast<unsigned>(
              output_resource.buffer != nullptr &&
              pipeline->device->ops->host_read(*pipeline->device,
                                               *output_resource.buffer)));
      break;
    }
    ++coherent_bank;
  }
  if (!coherent) {
    // Discrete Device-local Vulkan stays on the established rolling transfer
    // path until it has a distinct transfer-queue timeline DAG.
    const auto *const native =
        state->pipeline == nullptr || state->pipeline->device == nullptr
            ? nullptr
            : detail::accel_device(*state->pipeline->device);
    const auto *const adapter =
        native == nullptr
            ? nullptr
            : static_cast<const rund::node::accel::detail::VulkanAdapter *>(
                  native->pick.backend.context);
    std::fprintf(
        stderr, "vulkan product q=%zu coherent=%zu adapter=%u portability=%u\n",
        epochs, coherent_bank, static_cast<unsigned>(adapter != nullptr),
        static_cast<unsigned>(adapter != nullptr &&
                              adapter->portability_subset));
    return adapter != nullptr && !adapter->portability_subset;
  }
  detail::AccelDeviceState *const native =
      state->pipeline == nullptr || state->pipeline->device == nullptr
          ? nullptr
          : detail::accel_device(*state->pipeline->device);
  if (fault == ProductFault::KnownInput) {
    input_backing->fail_next_read();
  } else if (fault == ProductFault::DeviceLoss &&
             (native == nullptr ||
              !rund::node::accel::detail::InjectNativeDeviceLostOnce(
                  native->pick))) {
    return false;
  }
  const std::uint64_t before = ReadVulkanResidencyQueueSubmits(state->pipeline);
  const std::uint64_t output_writes_before = output_backing->writes();
  const auto started = std::chrono::steady_clock::now();
  const Status status = detail::run_virtual_pipeline(state);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const std::uint64_t after = ReadVulkanResidencyQueueSubmits(state->pipeline);
  const Stats stats = detail::virtual_pipeline_stats(state);
  // W<=4 lowers to the bounded one-submit window; W>4 lowers to the one-submit
  // arithmetic Schedule. Both are exactly one public/native queue handoff on
  // Vulkan. No chunk-count surrogate is accepted here.
  constexpr std::uint64_t queue_calls = 1u;
  constexpr std::uint64_t native_batches = 1u;
  if (fault == ProductFault::KnownInput) {
    const Status retry = detail::run_virtual_pipeline(state);
    const std::uint64_t retried =
        ReadVulkanResidencyQueueSubmits(state->pipeline);
    const Stats retry_stats = detail::virtual_pipeline_stats(state);
    const bool valid =
        status.reason() == Reason::BackendFailed && !status && retry &&
        after == before && retried == after + queue_calls &&
        stats.command_submits == 0u &&
        stats.pipeline.residency.epoch_count == 0u &&
        stats.pipeline.residency.window_handoff_count == 0u &&
        stats.pipeline.residency.window_batch_count == 0u &&
        stats.pipeline.residency.window_queue_call_count == 0u &&
        stats.publication.device_loss_count == 0u &&
        retry_stats.command_submits == queue_calls &&
        retry_stats.pipeline.residency.epoch_count == epochs &&
        retry_stats.pipeline.residency.window_handoff_count == 1u &&
        retry_stats.pipeline.residency.window_batch_count == native_batches &&
        retry_stats.pipeline.residency.window_queue_call_count == queue_calls &&
        input_backing->reads() == elements &&
        output_backing->writes() == elements && output_backing->all_u32(1u);
    if (!valid) {
      std::fprintf(stderr,
                   "vulkan known q=%zu status=%u retry=%u submits=%llu/%llu "
                   "window=%llu/%llu/%llu retry_window=%llu/%llu/%llu "
                   "queue=%llu/%llu/%llu reads=%llu writes=%llu output=%u\n",
                   epochs, static_cast<unsigned>(status.reason()),
                   static_cast<unsigned>(retry.reason()),
                   static_cast<unsigned long long>(stats.command_submits),
                   static_cast<unsigned long long>(retry_stats.command_submits),
                   static_cast<unsigned long long>(
                       stats.pipeline.residency.window_handoff_count),
                   static_cast<unsigned long long>(
                       stats.pipeline.residency.window_batch_count),
                   static_cast<unsigned long long>(
                       stats.pipeline.residency.window_queue_call_count),
                   static_cast<unsigned long long>(
                       retry_stats.pipeline.residency.window_handoff_count),
                   static_cast<unsigned long long>(
                       retry_stats.pipeline.residency.window_batch_count),
                   static_cast<unsigned long long>(
                       retry_stats.pipeline.residency.window_queue_call_count),
                   static_cast<unsigned long long>(before),
                   static_cast<unsigned long long>(after),
                   static_cast<unsigned long long>(retried),
                   static_cast<unsigned long long>(input_backing->reads()),
                   static_cast<unsigned long long>(output_backing->writes()),
                   static_cast<unsigned>(output_backing->all_u32(1u)));
    }
    return valid;
  }
  if (fault == ProductFault::DeviceLoss) {
    const Status retry = detail::run_virtual_pipeline(state);
    const std::uint64_t retried =
        ReadVulkanResidencyQueueSubmits(state->pipeline);
    const bool valid =
        status.reason() == Reason::DeviceLost &&
        retry.reason() == Reason::DeviceLost &&
        elapsed < std::chrono::seconds{2} && after == before + queue_calls &&
        retried == after && stats.command_submits == queue_calls &&
        stats.pipeline.residency.window_handoff_count == 1u &&
        stats.pipeline.residency.window_batch_count == native_batches &&
        stats.pipeline.residency.window_queue_call_count == queue_calls &&
        stats.publication.device_loss_count == 1u &&
        output_backing->writes() == output_writes_before;
    if (!valid) {
      std::fprintf(
          stderr,
          "vulkan loss q=%zu status=%u retry=%u elapsed_ms=%lld "
          "submits=%llu window=%llu/%llu/%llu loss=%llu queue=%llu/%llu/%llu\n",
          epochs, static_cast<unsigned>(status.reason()),
          static_cast<unsigned>(retry.reason()),
          static_cast<long long>(
              std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                  .count()),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(
              stats.pipeline.residency.window_handoff_count),
          static_cast<unsigned long long>(
              stats.pipeline.residency.window_batch_count),
          static_cast<unsigned long long>(
              stats.pipeline.residency.window_queue_call_count),
          static_cast<unsigned long long>(stats.publication.device_loss_count),
          static_cast<unsigned long long>(before),
          static_cast<unsigned long long>(after),
          static_cast<unsigned long long>(retried));
    }
    return valid;
  }
  const bool valid =
      status && stats.pipeline.residency.epoch_count == epochs &&
      stats.command_submits == queue_calls &&
      stats.pipeline.residency.window_handoff_count == 1u &&
      stats.pipeline.residency.window_batch_count == native_batches &&
      stats.pipeline.residency.window_queue_call_count == queue_calls &&
      stats.transfer_submissions.host_to_device == 0u &&
      stats.transfer_submissions.device_to_host == 0u &&
      after == before + queue_calls && output_backing->all_u32(1u) &&
      input_backing->reads() == elements &&
      output_backing->writes() == elements;
  if (!valid) {
    std::fprintf(
        stderr,
        "vulkan product q=%zu status=%u epochs=%llu window=%llu/%llu/%llu "
        "transfers=%llu/%llu "
        "submits=%llu/%llu reads=%llu writes=%llu output=%u\n",
        epochs, static_cast<unsigned>(status.reason()),
        static_cast<unsigned long long>(stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(
            stats.transfer_submissions.host_to_device),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_host),
        static_cast<unsigned long long>(before),
        static_cast<unsigned long long>(after),
        static_cast<unsigned long long>(input_backing->reads()),
        static_cast<unsigned long long>(output_backing->writes()),
        static_cast<unsigned>(output_backing->all_u32(1u)));
  }
  return valid;
}

} // namespace rund_node_test_pipeline

#endif
