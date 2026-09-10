#include "local.hpp"

namespace rund_node_test_pipeline {

AdmissionBacking::AdmissionBacking(const std::size_t bytes)
    : bytes_(bytes), fail_read_(false), reads_(0u), writes_(0u) {}

std::uint64_t AdmissionBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status
AdmissionBacking::read(const std::uint64_t offset,
                       const std::span<std::byte> output) noexcept {
  if (fail_read_.exchange(false, std::memory_order_acq_rel)) {
    return rund::compute::Status::fail(rund::compute::Reason::BackendFailed);
  }
  ++reads_;
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status
AdmissionBacking::write(const std::uint64_t offset,
                        const std::span<const std::byte> input) noexcept {
  ++writes_;
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  return rund::compute::Status::success();
}

std::uint64_t AdmissionBacking::callbacks() const noexcept {
  return reads_ + writes_;
}

void AdmissionBacking::fail_next_read() noexcept {
  fail_read_.store(true, std::memory_order_release);
}

bool AdmissionBacking::all_i32(const std::int32_t expected) const noexcept {
  if (bytes_.size() % sizeof(expected) != 0u) {
    return false;
  }
  for (std::size_t offset = 0u; offset < bytes_.size();
       offset += sizeof(expected)) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes_.data() + offset, sizeof(value));
    if (value != expected) {
      return false;
    }
  }
  return true;
}

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

ScheduleRouteScope::ScheduleRouteScope(
    rund::compute::detail::DeviceState &device) noexcept
    : device_(device), original_(device.ops), routed_{}, active_(false) {
  if (original_ == nullptr) {
    return;
  }
  routed_ = *original_;
  routed_.virtual_execution.admit_virtual_device_vsm = nullptr;
  routed_.virtual_execution.prepare_virtual_sliding_product = nullptr;
  routed_.virtual_execution.execute_virtual_sliding_product = nullptr;
  device_.ops = &routed_;
  active_ = true;
}

ScheduleRouteScope::~ScheduleRouteScope() {
  if (active_) {
    device_.ops = original_;
  }
}

ScheduleRouteScope::operator bool() const noexcept { return active_; }

NativeWait::NativeWait() noexcept : done(false), evidence{} {}

WindowWait::WindowWait() noexcept
    : done(false), valid(true), wrong_rejected(false), release_count(0u),
      context{}, pipelines{}, plan_identity(0u), token(0u), generation(0u),
      control_base(0u), suppressed(std::numeric_limits<std::size_t>::max()),
      terminal_loss(false), aborted(false), final{} {}

void CompleteNative(void *const raw,
                    execution::NativeEvidence &&evidence) noexcept {
  auto *const wait = static_cast<NativeWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->evidence = std::move(evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

void CompleteWindowRelease(
    void *const raw, rund::node::accel::detail::PreparedResidencyWindowRelease
                         &&release) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(release.receipt.epoch);
  const std::uint32_t expected =
      wait->control_base + static_cast<std::uint32_t>(index);
  const auto &evidence = release.evidence;
  const bool suppressed = index == wait->suppressed;
  const bool lost = wait->aborted || (wait->terminal_loss && index == 3u);
  const bool success = !suppressed && !lost;
  const auto terminal =
      lost ? rund::node::accel::detail::NativeTerminal::UnknownMayWrite
           : rund::node::accel::detail::NativeTerminal::Known;
  const bool valid =
      index < 4u && release.receipt.check.ok == success &&
      release.receipt.terminal == terminal &&
      release.receipt.backend_sequence == index + 1u &&
      release.receipt.bank == index % 2u && release.receipt.dispatched &&
      release.receipt.completed == !lost &&
      release.receipt.may_write == !suppressed &&
      evidence.check.ok == success && evidence.terminal == terminal &&
      evidence.submitted && evidence.control_observed == success &&
      evidence.control_valid == success &&
      (!success || evidence.control.generation == expected) &&
      evidence.active_step_count == 1u &&
      evidence.shared.run.work.command_submit_count == 1u;
  if (!valid) {
    wait->valid.store(false, std::memory_order_release);
  }
  wait->release_count.fetch_add(1u, std::memory_order_acq_rel);
  if (index < 2u && !wait->aborted) {
    const std::size_t next = index + 2u;
    rund::node::accel::detail::BackendResidencyWindowSignal signal{
        .plan_identity = wait->plan_identity,
        .token = wait->token,
        .generation = wait->generation,
        .epoch = next,
        .control_generation =
            wait->control_base + static_cast<std::uint32_t>(next),
        .bank = static_cast<std::uint8_t>(next % 2u),
    };
    if (next == wait->suppressed) {
      signal.admission = rund::AccelCheck{false, "compute_backend_failed"};
    }
    if (index == 0u) {
      auto wrong = signal;
      ++wrong.control_generation;
      const rund::AccelCheck rejected =
          rund::node::accel::detail::SignalPreparedKernelPipelineWindow(
              wait->context, wait->pipelines[next % 2u], wrong);
      wait->wrong_rejected.store(!rejected.ok, std::memory_order_release);
    }
    const rund::AccelCheck signaled =
        rund::node::accel::detail::SignalPreparedKernelPipelineWindow(
            wait->context, wait->pipelines[next % 2u], signal);
    if (!signaled.ok) {
      wait->valid.store(false, std::memory_order_release);
    }
  }
}

void CompleteWindowFinal(
    void *const raw,
    rund::node::accel::detail::PreparedResidencyWindowFinal &&final) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->final = std::move(final.evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

[[nodiscard]] rund::AccelCheck RejectPipelineExecution(
    const rund::compute::detail::DeviceState &,
    const rund::node::accel::detail::PreparedKernelPipeline &,
    const std::span<const std::uint32_t>, std::shared_ptr<void>,
    const rund::node::accel::detail::PreparedPipelineCompletion,
    void *) noexcept {
  return {false, "compute_backend_failed"};
}

[[nodiscard]] rund::AccelCheck TamperPipelineExecution(
    const rund::compute::detail::DeviceState &,
    const rund::node::accel::detail::PreparedKernelPipeline &,
    const std::span<const std::uint32_t> locals, std::shared_ptr<void> lifetime,
    const rund::node::accel::detail::PreparedPipelineCompletion completion,
    void *const user) noexcept {
  auto state =
      std::static_pointer_cast<rund::compute::detail::PipelineState>(lifetime);
  if (state == nullptr || completion == nullptr || locals.empty()) {
    return {false, "compute_pipeline_invalid"};
  }
  rund::node::accel::detail::PreparedPipelineEvidence evidence{};
  evidence.check = {true, "ok"};
  evidence.shared.outcome = {.ok = true, .reason = "ok"};
  evidence.shared.run.work.command_submit_count = 1u;
  evidence.shared.run.work.command_inflight_peak = 1u;
  evidence.active_step_count = static_cast<std::uint32_t>(locals.size());
  evidence.submitted = true;
  evidence.control_observed = true;
  evidence.control_valid = true;
  evidence.control_byte_count =
      rund::node::accel::detail::PreparedPipelineControlBytes;
  evidence.control.generation =
      static_cast<std::uint32_t>(state->attempt.generation + 2u);
  evidence.control.reason =
      static_cast<std::uint32_t>(rund::compute::Reason::Ok);
  evidence.control.failed_step =
      rund::node::accel::detail::PreparedPipelineNoStep;
  evidence.control.verified_prefix = static_cast<std::uint32_t>(locals.size());
  auto duplicate = evidence;
  completion(user, std::move(evidence));
  completion(user, std::move(duplicate));
  return {true, "ok"};
}

[[nodiscard]] execution::SealResult
ExecutionPlan(const residency::Pool &pool,
              const std::uint64_t page_count) noexcept {
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
                      .page_bytes = sizeof(std::int32_t),
                      .page_count = page_count,
                  },
              .access = residency::Access::Read,
              .logical_bytes = page_count * sizeof(std::int32_t),
              .payload_bytes = sizeof(std::int32_t),
              .next_use_base = page_count,
          },
      .output =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 2u,
                      .key = residency::CacheKey{.backing = 2u, .version = 1u},
                      .page_bytes = sizeof(std::int32_t),
                      .page_count = page_count,
                  },
              .access = residency::Access::Write,
              .logical_bytes = page_count * sizeof(std::int32_t),
              .payload_bytes = sizeof(std::int32_t),
          },
      .publication =
          execution::Publication{
              .backing = 2u,
              .version = 1u,
              .extent =
                  residency::DirtyExtent{
                      .offset = 0u, .bytes = page_count * sizeof(std::int32_t)},
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

[[nodiscard]] bool SameCurrent(const rund::compute::MemoryStats &left,
                               const rund::compute::MemoryStats &right) {
  return left.host.current == right.host.current &&
         left.frame.current == right.frame.current &&
         left.tile.current == right.tile.current &&
         left.resident.current == right.resident.current &&
         left.staging.current == right.staging.current &&
         left.device.current == right.device.current &&
         left.transfer.current == right.transfer.current;
}

[[nodiscard]] bool
AddedSubmissionBytes(const rund::compute::PipelinePlan &before,
                     const rund::compute::PipelinePlan &after,
                     const std::uint64_t bytes) {
  const auto added = [bytes](const std::uint64_t left,
                             const std::uint64_t right) {
    return left <= std::numeric_limits<std::uint64_t>::max() - bytes &&
           right == left + bytes;
  };
  return added(before.prepared_native_bytes, after.prepared_native_bytes) &&
         added(before.prepared_bytes, after.prepared_bytes) &&
         added(before.peak_bytes, after.peak_bytes) &&
         added(before.committed_peak_bytes, after.committed_peak_bytes) &&
         added(before.total_bytes, after.total_bytes) &&
         added(before.logical_bytes, after.logical_bytes) &&
         added(before.live_bytes, after.live_bytes) &&
         added(before.physical_bytes, after.physical_bytes);
}

#endif

} // namespace rund_node_test_pipeline
