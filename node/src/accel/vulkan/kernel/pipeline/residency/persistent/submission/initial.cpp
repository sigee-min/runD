#include "../internal.hpp"

#include <cstring>
#include <new>
#include <vector>

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

PersistentResidencySlidingSubmitResult
submit_initial(const PersistentResidencySlidingRequest &request,
               PersistentResidencySlidingControl &control,
               const PersistentResidencySlidingCapability &capability,
               Owner &owner, VulkanAdapter &adapter,
               VulkanResidencyPersistentRun &run) noexcept {
  std::array<VulkanResidencyPersistentRole, ResidencySlidingCapacity> roles{};
  VulkanResidencyPersistentRole tail{};
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(request.roles[slot].prepared.get());
    if (!ValidVulkanPipeline(pipeline) || pipeline->residency == nullptr ||
        pipeline->adapter != &adapter) {
      clear_attempt_claims(request, &run);
      destroy_attempt_roles(request);
      abort_rearm(owner);
      reset_control_unlocked(control);
      return submit_out(invalid());
    }
    VulkanResidencyPersistentRun *expected = nullptr;
    if (!pipeline->residency->active_persistent.compare_exchange_strong(
            expected, &run, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      clear_attempt_claims(request, &run);
      destroy_attempt_roles(request);
      abort_rearm(owner);
      reset_control_unlocked(control);
      return submit_out({false, "compute_pipeline_busy"});
    }
  }
  VulkanAdapter *materialized_adapter = &adapter;
  VulkanResidencyPersistentRun *materialized_run = &run;
  if (!materialize(request, roles, tail, materialized_adapter, materialized_run,
                   &run) ||
      materialized_run != &run || materialized_adapter != &adapter) {
    clear_attempt_claims(request, &run);
    destroy_attempt_roles(request);
    abort_rearm(owner);
    reset_control_unlocked(control);
    return submit_out(invalid());
  }
  run.request = request;
  run.control = &control;
  run.roles = roles;
  run.tail = tail;
  run.cells = {};
  run.submit_begin_ns = MonotonicNanoseconds();
  run.active = false;
  run.final_sent = false;
  const std::uint64_t first = request.first_coordinate;
  const std::uint64_t span =
      request.mode == PersistentResidencySlidingMode::OneSubmit
          ? request.coordinate_count
          : request.chunk_count;
  if (first >= request.coordinate_count || span == 0u ||
      span > request.coordinate_count - first ||
      (request.mode == PersistentResidencySlidingMode::BackendChunked &&
       !persistent_sliding_chunk_valid(capability, request))) {
    clear_attempt_claims(request, &run);
    destroy_attempt_roles(request);
    abort_rearm(owner);
    reset_control_unlocked(control);
    return submit_out(invalid());
  }
  std::uint32_t generation = 0u;
  rund::AccelCheck prepared =
      ReserveVulkanTimelineGeneration(adapter.timeline, generation);
  const bool chunked =
      request.mode == PersistentResidencySlidingMode::BackendChunked;
  std::array<VulkanTimelineBatch, BatchCount> chunk_batches{};
  std::vector<VulkanTimelineBatch> whole_batches{};
  if (!chunked) {
    try {
      whole_batches.resize(static_cast<std::size_t>(span));
    } catch (const std::bad_alloc &) {
      if (generation != 0u) {
        static_cast<void>(
            CancelVulkanTimelineGeneration(adapter.timeline, generation));
      }
      clear_attempt_claims(request, &run);
      run.request = {};
      run.control = nullptr;
      destroy_attempt_roles(request);
      abort_rearm(owner);
      reset_control_unlocked(control);
      return submit_out({false, "compute_pipeline_capacity"});
    }
  }
  std::array<std::uint64_t, VulkanResidencyWindowCapacity> ready_base{};
  std::uint64_t done_base = 0u;
  for (std::uint64_t coordinate = first;
       prepared.ok && coordinate < first + span; ++coordinate) {
    VulkanTimelinePoint current{};
    prepared =
        ReserveVulkanTimelinePoint(adapter.timeline, generation, current);
    if (!prepared.ok) {
      break;
    }
    const std::size_t cell =
        static_cast<std::size_t>(coordinate % VulkanResidencyWindowCapacity);
    if (coordinate < VulkanResidencyWindowCapacity) {
      ready_base[cell] = current.ready_value;
    }
    if (coordinate == 0u) {
      done_base = current.value;
    }
    VulkanResidencyPersistentRole &native = native_role(run, coordinate);
    const VulkanTimelineBatch batch{
        .commands = std::span<const VkCommandBuffer>{native.commands.data(),
                                                     native.command_count},
        .point = current,
    };
    if (chunked) {
      owner.batches[static_cast<std::size_t>(coordinate % BatchCount)] = batch;
      chunk_batches[static_cast<std::size_t>(coordinate - first)] = batch;
    } else {
      whole_batches[static_cast<std::size_t>(coordinate)] = batch;
    }
  }
  if (!prepared.ok) {
    if (generation != 0u) {
      static_cast<void>(
          CancelVulkanTimelineGeneration(adapter.timeline, generation));
    }
    clear_attempt_claims(request, &run);
    run.request = {};
    run.control = nullptr;
    destroy_attempt_roles(request);
    abort_rearm(owner);
    reset_control_unlocked(control);
    return submit_out(prepared);
  }
  std::uint64_t queue_calls = 0u;
  const std::span<const VulkanTimelineBatch> batches =
      chunked
          ? std::span<const VulkanTimelineBatch>{chunk_batches.data(),
                                                 static_cast<std::size_t>(span)}
          : std::span<const VulkanTimelineBatch>{whole_batches.data(),
                                                 whole_batches.size()};
  const rund::AccelCheck submitted =
      SubmitVulkanTimelineStream(adapter, batches, queue_calls, true);
  const PersistentResidencySlidingRequest pending = owner.pending_request;
  const bool committed =
      submitted.ok && queue_calls == 1u && commit_rearm(owner);
  if (submitted.ok && queue_calls == 1u &&
      (!committed || owner.prepared.mode != request.mode ||
       !same_prepared_request(owner, request))) {
    owner.prepared = committed ? request : pending;
    owner.prepared.lowering.reset();
    owner.owner_nonce = request.owner_nonce;
    run.ready_base = ready_base;
    run.done_base = done_base;
    run.timeline_generation = generation;
    run.active = true;
    quarantine_unknown_locked(control, &owner, &run, &adapter,
                              rund::AccelCheck{false, "compute_device_lost"});
    return submit_out({false, "compute_device_lost"},
                      PersistentResidencySlidingSubmitEvent::AcceptedUnknown);
  }
  if (!submitted.ok || queue_calls != 1u) {
    const bool device_lost =
        !submitted.ok && submitted.reason != nullptr &&
        std::strcmp(submitted.reason, "compute_device_lost") == 0;
    // A nonzero queue count means native work may have escaped even when the
    // API reports a failed or malformed submission. Keep the owner and
    // control live for the common accepted-prefix terminal path; only a
    // genuinely pre-accept attempt may be reset below.
    if (device_lost || queue_calls != 0u) {
      run.ready_base = ready_base;
      run.done_base = done_base;
      run.timeline_generation = generation;
      run.active = true;
      quarantine_unknown_locked(
          control, &owner, &run, &adapter,
          submitted.ok ? rund::AccelCheck{false, "compute_device_lost"}
                       : submitted);
      return submit_out({false, "compute_device_lost"},
                        PersistentResidencySlidingSubmitEvent::AcceptedUnknown);
    }
    static_cast<void>(
        CancelVulkanTimelineGeneration(adapter.timeline, generation));
    clear_attempt_claims(request, &run);
    run.request = {};
    run.control = nullptr;
    destroy_attempt_roles(request);
    abort_rearm(owner);
    reset_control_unlocked(control);
    return submit_out(submitted.ok ? unavailable() : submitted);
  }
  run.ready_base = ready_base;
  run.done_base = done_base;
  run.timeline_generation = generation;
  run.active = true;
  return submit_out({true, "ok"},
                    PersistentResidencySlidingSubmitEvent::Accepted);
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
