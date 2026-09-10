#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstring>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline::vulkan_transfer {

VulkanTransferBacking::VulkanTransferBacking(const std::size_t bytes)
    : bytes_(bytes) {}

std::uint64_t VulkanTransferBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status VulkanTransferBacking::read(
    const std::uint64_t offset,
    const std::span<std::byte> output) noexcept {
  ++callbacks_;
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(
        rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status VulkanTransferBacking::write(
    const std::uint64_t offset,
    const std::span<const std::byte> input) noexcept {
  ++callbacks_;
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(
        rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  return rund::compute::Status::success();
}

std::uint64_t VulkanTransferBacking::callbacks() const noexcept {
  return callbacks_;
}

void CompleteVulkanExecution(
    void *const raw, execution::NativeEvidence &&evidence) noexcept {
  auto *const wait = static_cast<VulkanExecutionWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->evidence = std::move(evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

execution::SealResult VulkanExecutionPlan(
    const residency::Pool &pool, const std::uint64_t page_count) noexcept {
  const std::uint32_t frames = pool.layout.frame_capacity;
  const auto region = [frames](const residency::FrameTier tier,
                               const residency::FrameRole role,
                               const std::uint32_t first) noexcept {
    return residency::FrameRegion{
        .tier = tier, .role = role, .first = first, .count = frames};
  };
  return execution::seal(execution::Request{
      .page_count = page_count,
      .frame_capacity = frames,
      .input =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 1u,
                      .key = residency::CacheKey{.backing = 1u, .version = 1u},
                      .page_bytes = sizeof(std::uint32_t),
                      .page_count = page_count,
                  },
              .access = residency::Access::Read,
              .logical_bytes = page_count * sizeof(std::uint32_t),
              .payload_bytes = sizeof(std::uint32_t),
              .next_use_base = page_count,
          },
      .output =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 2u,
                      .key = residency::CacheKey{.backing = 2u, .version = 1u},
                      .page_bytes = sizeof(std::uint32_t),
                      .page_count = page_count,
                  },
              .access = residency::Access::Write,
              .logical_bytes = page_count * sizeof(std::uint32_t),
              .payload_bytes = sizeof(std::uint32_t),
          },
      .publication =
          execution::Publication{
              .backing = 2u,
              .version = 1u,
              .extent = residency::DirtyExtent{.offset = 0u,
                                               .bytes = page_count *
                                                        sizeof(std::uint32_t)},
          },
      .host_input = {region(residency::FrameTier::Host,
                            residency::FrameRole::Input,
                            pool.first_host_input_frame),
                     region(residency::FrameTier::Host,
                            residency::FrameRole::Input,
                            pool.first_host_input_frame +
                                pool.layout.host_frame_capacity)},
      .device_input = pool.input_regions,
      .device_output = {region(residency::FrameTier::Device,
                               residency::FrameRole::Output,
                               pool.first_output_frame),
                        region(residency::FrameTier::Device,
                               residency::FrameRole::Output,
                               pool.first_output_frame + frames)},
      .host_output = {region(residency::FrameTier::Host,
                             residency::FrameRole::Output,
                             pool.first_host_output_frame),
                      region(residency::FrameTier::Host,
                             residency::FrameRole::Output,
                             pool.first_host_output_frame + frames)},
  });
}

void CompleteVulkanSelection(
    void *const raw,
    rund::node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept {
  auto *const wait = static_cast<VulkanSelectionWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->evidence = std::move(evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

bool RunVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const std::span<const std::uint32_t> locals,
    rund::node::accel::detail::PreparedPipelineEvidence &evidence) {
  using namespace rund::node::accel::detail;
  if (state == nullptr || state->device == nullptr) {
    return false;
  }
  const auto *const accelerator =
      std::get_if<rund::compute::detail::AccelDeviceState>(
          &state->device->storage);
  if (accelerator == nullptr) {
    return false;
  }
  VulkanSelectionWait wait{};
  const rund::AccelCheck submitted = SubmitPreparedKernelPipelineSelection(
      accelerator->context, state->prepared, state, CompleteVulkanSelection,
      &wait, KernelTiming::None, PipelineSubmitMode::Residency, locals);
  if (!submitted.ok) {
    return false;
  }
  wait.done.wait(false, std::memory_order_acquire);
  evidence = std::move(wait.evidence);
  return evidence.check.ok && evidence.terminal == NativeTerminal::Known &&
         evidence.submitted && evidence.control_observed &&
         evidence.control_valid &&
         evidence.active_step_count == locals.size() &&
         evidence.control.verified_prefix == locals.size() &&
         evidence.shared.run.work.command_submit_count == 1u &&
         evidence.shared.run.work.command_capacity == 1u &&
         evidence.shared.run.work.command_inflight_peak == 1u;
}

bool SeedVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  return state != nullptr &&
         rund::node::accel::detail::SeedPreparedKernelPipelineGeneration(
             state->prepared, 0u)
             .ok;
}

std::uint64_t VulkanQueueSubmits(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  using namespace rund::node::accel::detail;
  if (state == nullptr || !state->prepared.ok) {
    return 0u;
  }
  auto *const prepared =
      static_cast<prepared::PipelineState *>(state->prepared.owner.get());
  auto *const pipeline =
      prepared == nullptr
          ? nullptr
          : static_cast<VulkanPipeline *>(prepared->backend.get());
  if (pipeline == nullptr || pipeline->adapter == nullptr) {
    return 0u;
  }
  std::lock_guard lock{pipeline->adapter->mutex};
  return pipeline->adapter->command_submit_count;
}

bool DownloadVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const rund::compute::Buffer<std::uint32_t> &buffer,
    std::array<std::uint32_t, 8u> &output) {
  using namespace rund::compute::detail;
  const std::shared_ptr<BufferState> &storage = BufferAccess::state(buffer);
  if (state == nullptr || state->device == nullptr || storage == nullptr ||
      storage->device != state->device || state->device->ops == nullptr ||
      state->device->ops->download == nullptr) {
    return false;
  }
  const DownloadResult downloaded = state->device->ops->download(
      *state->device, *storage, output.data(), sizeof(output), 0u);
  return static_cast<bool>(downloaded.status) && downloaded.payload_hash_valid;
}

bool CheckVulkanRangeFailure(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const std::shared_ptr<rund::compute::detail::BufferState> &buffer,
    const std::array<std::uint32_t, 8u> &expected) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  if (state == nullptr || state->device == nullptr || buffer == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->download == nullptr ||
      state->device->ops->download_batch == nullptr) {
    return false;
  }
  std::array<std::uint32_t, 2u> shifted{};
  std::uint64_t shifted_hash = 0u;
  rund::node::accel::detail::DownloadRangeOutcome shifted_outcome{};
  const DownloadRequest shifted_request{
      .buffer = buffer.get(),
      .data = shifted.data(),
      .bytes = sizeof(shifted),
      .offset = sizeof(std::uint32_t),
      .payload_hash = &shifted_hash,
      .outcome = &shifted_outcome,
  };
  const DownloadResult shifted_result = state->device->ops->download_batch(
      *state->device, std::span<const DownloadRequest>{&shifted_request, 1u},
      rund::node::accel::detail::TransferAuthority::Shared);
  if (!shifted_result.status || shifted_result.command_submits != 1u ||
      shifted_outcome.state !=
          rund::node::accel::detail::DownloadRangeState::Complete ||
      shifted_outcome.confirmed_bytes != sizeof(shifted) ||
      !shifted_outcome.hash_valid || shifted[0u] != expected[1u] ||
      shifted[1u] != expected[2u]) {
    return false;
  }

  std::array<std::uint32_t, 2u> scalar{};
  const DownloadResult scalar_result =
      state->device->ops->download(*state->device, *buffer, scalar.data(),
                                   sizeof(scalar), sizeof(std::uint32_t));
  if (!scalar_result.status || scalar_result.command_submits != 1u ||
      !scalar_result.payload_hash_valid ||
      scalar_result.payload_hash != shifted_hash ||
      scalar[0u] != expected[1u] || scalar[1u] != expected[2u]) {
    return false;
  }

  std::array<std::uint32_t, 2u> first{};
  std::array<std::uint32_t, 2u> second{};
  std::uint64_t first_hash = 0u;
  std::uint64_t second_hash = 0u;
  rund::node::accel::detail::DownloadRangeOutcome outcomes[2u]{};
  const DownloadRequest requests[2u]{
      {.buffer = buffer.get(),
       .data = first.data(),
       .bytes = sizeof(first),
       .offset = 0u,
       .payload_hash = &first_hash,
       .outcome = &outcomes[0u]},
      {.buffer = buffer.get(),
       .data = second.data(),
       .bytes = sizeof(second),
       .offset = 2u * sizeof(std::uint32_t),
       .payload_hash = &second_hash,
       .outcome = &outcomes[1u]},
  };
  const auto *const accelerator =
      std::get_if<AccelDeviceState>(&state->device->storage);
  if (accelerator == nullptr ||
      !rund::node::accel::detail::InjectNativeTransferDeviceLostOnce(
          accelerator->pick)) {
    return false;
  }
  const DownloadResult failed = state->device->ops->download_batch(
      *state->device, std::span<const DownloadRequest>{requests, 2u},
      rund::node::accel::detail::TransferAuthority::Shared);
  return !failed.status && failed.status.reason() == Reason::DeviceLost &&
         failed.command_submits == 1u && failed.first_failed_valid &&
         failed.first_failed == 0u && failed.confirmed_bytes == 0u &&
         failed.ordered_prefix == 0u &&
         outcomes[0u].state ==
             rund::node::accel::detail::DownloadRangeState::FailedNoWrite &&
         outcomes[0u].confirmed_bytes == 0u &&
         outcomes[1u].state ==
             rund::node::accel::detail::DownloadRangeState::Untouched &&
         outcomes[1u].confirmed_bytes == 0u;
}

bool RejectVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const std::span<const std::uint32_t> locals) {
  using namespace rund::node::accel::detail;
  if (state == nullptr || state->device == nullptr) {
    return false;
  }
  const auto *const accelerator =
      std::get_if<rund::compute::detail::AccelDeviceState>(
          &state->device->storage);
  if (accelerator == nullptr ||
      !SeedPreparedKernelPipelineGeneration(state->prepared, 0u)) {
    return false;
  }
  VulkanSelectionWait wait{};
  const std::uint64_t before = VulkanQueueSubmits(state);
  const rund::AccelCheck rejected = SubmitPreparedKernelPipelineSelection(
      accelerator->context, state->prepared, state, CompleteVulkanSelection,
      &wait, KernelTiming::None, PipelineSubmitMode::Residency, locals);
  return !rejected.ok && !wait.done.load(std::memory_order_acquire) &&
         VulkanQueueSubmits(state) == before;
}

bool SameCurrentCounter(const rund::compute::MemoryCounter &left,
                        const rund::compute::MemoryCounter &right) {
  // A rejected constructor may allocate then release cold candidates. Peak,
  // cumulative, and reuse remain evidence of that real producer activity;
  // rollback requires that no retained owner or budget authority survives.
  return left.current == right.current && left.budget == right.budget;
}

bool SameDeviceMemoryAuthority(const rund::compute::MemoryStats &left,
                               const rund::compute::MemoryStats &right) {
  return left.backend == right.backend && left.scope == right.scope &&
         SameCurrentCounter(left.host, right.host) &&
         SameCurrentCounter(left.device, right.device) &&
         SameCurrentCounter(left.frame, right.frame) &&
         SameCurrentCounter(left.tile, right.tile) &&
         SameCurrentCounter(left.resident, right.resident) &&
         SameCurrentCounter(left.staging, right.staging) &&
         SameCurrentCounter(left.transfer, right.transfer);
}

} // namespace rund_node_test_pipeline::vulkan_transfer

#endif
