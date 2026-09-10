#include "../readiness.hpp"
#include "internal.hpp"

#include <mutex>

namespace rund::compute::detail {

namespace {

void cleanup_sliding_preparation(std::shared_ptr<void> &raw,
                                 const std::uint64_t lease_token) noexcept {
  const auto owner =
      std::static_pointer_cast<sliding_product_detail::SlidingProductOwner>(
          raw);
  if (owner == nullptr || owner->run == nullptr) {
    return;
  }
  auto &preparation = owner->run->persistent_preparation.backend;
  const bool retained =
      (preparation.ticket != nullptr && preparation.ticket->quarantined) ||
      owner->run->quarantine != nullptr ||
      owner->run->poison.load(std::memory_order_acquire);
  const bool preinit = owner->run->cold_owner == nullptr;
  if (!retained && preinit && preparation.pending_ticket() &&
      preparation.ticket != nullptr) {
    node::accel::detail::persistent_sliding_abort_ticket(preparation.ticket);
  }
  if (!retained && preinit && owner->pending_valid) {
    sliding_product_detail::discard_staged_roles(*owner);
  }
  if (!retained && lease_token != 0u && owner->prepared_lease_count != 0u &&
      owner->prepared_lease_token == lease_token) {
    owner->prepared_lease_count = 0u;
  }
}

[[nodiscard]] bool
acquire_prepared_lease(sliding_product_detail::SlidingProductOwner &owner,
                       std::uint64_t &lease_token) noexcept {
  if (owner.prepared_lease_count != 0u ||
      owner.prepared_lease_token == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  ++owner.prepared_lease_token;
  if (owner.prepared_lease_token == 0u) {
    return false;
  }
  owner.prepared_lease_count = 1u;
  lease_token = owner.prepared_lease_token;
  return true;
}

[[nodiscard]] bool cache_idle_locked(
    const sliding_product_detail::SlidingProductOwner &owner) noexcept {
  if (owner.run == nullptr || owner.memory == nullptr ||
      !owner.run->persistent_preparation) {
    return false;
  }
  sliding_product_detail::SlidingProductRun &run = *owner.run;
  if (run.cold_owner != nullptr || run.quarantine != nullptr ||
      owner.pending_valid ||
      run.persistent_preparation.backend.pending_ticket() ||
      (run.persistent_preparation.backend.ticket != nullptr &&
       run.persistent_preparation.backend.ticket->quarantined) ||
      owner.prepared_lease_count != 0u ||
      run.poison.load(std::memory_order_acquire)) {
    return false;
  }
  {
    std::lock_guard control_lock{run.persistent_control.gate};
    if (run.persistent_control.active ||
        run.persistent_control.native != nullptr) {
      return false;
    }
    if (!run.sliding.quiescent()) {
      return false;
    }
    for (std::size_t bank = 0u; bank < run.pipeline_started.size(); ++bank) {
      if (run.pipeline_started[bank] || run.pipeline_submitted[bank]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool
make_key(const VirtualPipelineState &state, const VirtualRunProjection &run,
         const residency::execution::Plan &plan,
         sliding_product_detail::SlidingProductKey &key) noexcept {
  if (state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.alternate_pipeline->device != state.pipeline->device ||
      run.input_count > VirtualPipelineState::InputCapacity) {
    return false;
  }
  key = {};
  key.owners = std::array<const void *, 4u>{
      state.pipeline->device.get(), state.pipeline->device->ops,
      state.pipeline.get(), state.alternate_pipeline.get()};
  key.host_input = plan.host_input_regions();
  key.device_input = plan.device_input_regions();
  key.device_output = plan.device_output_regions();
  key.host_output = plan.host_output_regions();
  key.memory = node::accel::detail::ResidencySlidingMemory::HostCoherent;

  std::size_t count = 0u;
  const auto add = [&](const std::uint64_t value) noexcept {
    if (count >= key.scalar.size()) {
      return false;
    }
    key.scalar[count++] = value;
    return true;
  };
  const auto add_bool = [&](const bool value) noexcept {
    return add(value ? 1u : 0u);
  };
  const VirtualBufferState *const input = virtual_input(state);
  if (input == nullptr || input->backing == nullptr) {
    return false;
  }
  if (!add(plan.identity()) || !add(plan.page_count()) ||
      !add(plan.frame_capacity()) || !add(plan.epoch_count()) ||
      !add(plan.input_resource()) || !add(plan.output_resource()) ||
      !add(residency::execution::BankCapacity) ||
      !add(node::accel::detail::PersistentResidencySlidingCapacity) ||
      !add(static_cast<std::uint8_t>(state.geometry.route)) ||
      !add(static_cast<std::uint8_t>(run.input_type)) ||
      !add(static_cast<std::uint8_t>(run.output_type)) ||
      !add(state.geometry.operation) || !add(state.geometry.boundary) ||
      !add(state.geometry.input_payload_elements) ||
      !add(state.geometry.output_payload_elements) ||
      !add(state.geometry.input_frame_elements) ||
      !add(state.geometry.output_frame_elements) ||
      !add(state.geometry.intermediate_frame_elements) ||
      !add(state.geometry.input_prefix_elements) ||
      !add(state.geometry.output_prefix_elements) ||
      !add(state.geometry.materialization_hi) ||
      !add(state.geometry.materialization_lo) || !add(run.input_count) ||
      !add(run.input_capacity_bytes) || !add(run.input_identity_hi) ||
      !add(run.canonical_input_identity_hi) ||
      !add(run.canonical_input_identity_lo) || !add(run.result_identity_hi) ||
      !add(run.result_identity_lo) || !add(run.input_backing) ||
      !add(run.output_backing) || !add(run.input_page_bytes) ||
      !add(run.intermediate_page_bytes) || !add(run.control_page_bytes) ||
      !add(run.output_page_bytes) || !add(run.input_payload_bytes) ||
      !add(run.output_payload_bytes) || !add(run.input_prefix_bytes) ||
      !add(run.output_prefix_bytes) || !add(run.input_frame_elements) ||
      !add(run.frame_capacity) || !add(run.input_arena_bytes) ||
      !add(run.intermediate_arena_bytes) || !add(run.control_arena_bytes) ||
      !add(run.output_arena_bytes) || !add(run.first_intermediate_frame) ||
      !add(run.first_output_frame) || !add(run.first_host_input_frame) ||
      !add(run.host_input_count) || !add(run.host_frame_capacity) ||
      !add(run.first_host_output_frame) ||
      !add(run.host_output_frame_capacity) || !add(run.graph_resource_count) ||
      !add_bool(run.clamp_window) || !add_bool(run.clip_window) ||
      !add_bool(run.device_vsm_required) || !add_bool(run.reduction()) ||
      !add_bool(run.graph_execution()) || !add_bool(run.graph_reduction()) ||
      !add_bool(run.scan()) || !add_bool(run.inclusive_scan()) ||
      !add_bool(run.multi_pointwise()) || !add_bool(run.multi_scan())) {
    return false;
  }
  if (!add(static_cast<std::uint8_t>(input->backing->tier())) ||
      !add(input->backing->max_parallel_reads())) {
    return false;
  }
  for (std::size_t index = 0u; index < VirtualPipelineState::InputCapacity;
       ++index) {
    const VirtualRunInputProjection &input = run.inputs[index];
    if (!add(input.capacity_bytes) || !add(input.identity_hi) ||
        !add(input.identity_lo) || !add(input.backing) ||
        !add(static_cast<std::uint8_t>(input.type))) {
      return false;
    }
  }
  key.scalar_count = count;
  return true;
}

} // namespace

Status prepare_accel_virtual_execution_sliding_locked(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    VirtualExecutionSlidingPrepared &prepared) noexcept {
  using namespace sliding_product_detail;
  prepared.reset();
  residency::execution::SealResult sealed{};
  node::accel::detail::PersistentResidencySlidingMode mode =
      node::accel::detail::PersistentResidencySlidingMode::OneSubmit;
  if (!admit_preparation(state, run, sealed, mode)) {
    return Status::fail(Reason::BackendUnsupported);
  }
  Status status = ready_virtual_residency_window(state);
  if (!status) {
    return status;
  }

  const std::array<std::shared_ptr<PipelineState>,
                   residency::execution::BankCapacity>
      pipelines{state.pipeline, state.alternate_pipeline};
  PipelineExecutionSnapshot snapshot{};
  status = snapshot_pipeline_execution(pipelines, snapshot);
  if (!status) {
    return status;
  }
  sliding_product_detail::SlidingProductKey key{};
  if (!make_key(state, run, sealed.plan, key)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::shared_ptr<SlidingProductOwner> cached{};
  cached = state.sliding_product_cache;
  const bool same =
      cached != nullptr && cached->key_sealed && cached->key == key;
  if (same && cached->run != nullptr) {
    std::lock_guard run_lock{cached->run->gate};
    if (cached->mode != mode) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (cache_idle_locked(*cached)) {
      status = stage_preparation_roles(*cached, snapshot);
      if (!status) {
        return status;
      }
      status = verify_persistent_backend(*cached);
      if (!status) {
        discard_staged_roles(*cached);
        return status;
      }
      std::uint64_t lease_token = 0u;
      if (!acquire_prepared_lease(*cached, lease_token)) {
        discard_staged_roles(*cached);
        return Status::fail(Reason::PipelineBusy);
      }
      prepared.plan = cached->plan;
      prepared.owner = cached;
      prepared.cleanup = cleanup_sliding_preparation;
      prepared.lease_token = lease_token;
      return Status::success();
    }
    if (cached->run->quarantine != nullptr ||
        cached->run->poison.load(std::memory_order_acquire)) {
      return Status::fail(Reason::DeviceLost);
    }
    return Status::fail(Reason::PipelineBusy);
  }
  if (cached != nullptr) {
    if (cached->run == nullptr) {
      return Status::fail(Reason::PipelineBusy);
    }
    std::lock_guard run_lock{cached->run->gate};
    if (cached->run->quarantine != nullptr ||
        cached->run->poison.load(std::memory_order_acquire)) {
      return Status::fail(Reason::DeviceLost);
    }
    if (!cache_idle_locked(*cached)) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (state.sliding_product_cache == cached) {
      state.sliding_product_cache.reset();
    }
  }
  cached.reset();

  storage::Reservation capacity{};
  status = reserve_persistent_capacity(state, capacity);
  if (!status) {
    return status;
  }
  std::shared_ptr<SlidingProductOwner> owner{};
  status = allocate_preparation_owner(sealed.plan, std::move(capacity), owner);
  if (status) {
    status = snapshot_preparation_pipelines(state, *owner);
  }
  if (status) {
    status = build_preparation_roles(*owner);
  }
  if (status) {
    owner->mode = mode;
  }
  if (status) {
    status = verify_persistent_backend(*owner);
  }
  if (!status) {
    return status;
  }
  owner->key = key;
  owner->key_sealed = true;
  std::uint64_t lease_token = 0u;
  if (!acquire_prepared_lease(*owner, lease_token)) {
    return Status::fail(Reason::PipelineBusy);
  }
  state.sliding_product_cache = owner;
  prepared.plan = owner->plan;
  prepared.owner = std::move(owner);
  prepared.cleanup = cleanup_sliding_preparation;
  prepared.lease_token = lease_token;
  return Status::success();
}

Status prepare_accel_virtual_execution_sliding(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    VirtualExecutionSlidingPrepared &prepared) noexcept {
  std::unique_lock lock{state.gate, std::try_to_lock};
  if (!lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  return prepare_accel_virtual_execution_sliding_locked(state, run, prepared);
}

} // namespace rund::compute::detail
