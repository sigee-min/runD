#include "local.hpp"

#include <chrono>
#include <utility>

namespace rund_node_test_pipeline_residency::native_sliding {

[[nodiscard]] accel::PreparedResidencySlidingProjection
ProjectSliding(void *const raw, const std::uint64_t coordinate,
               const std::uint64_t turn, const std::uint8_t slot,
               accel::PreparedResidencySlidingSelection &selection) noexcept {
  auto *const wait = static_cast<SlidingWait *>(raw);
  if (wait == nullptr || wait->stride == 0u ||
      wait->stride > wait->slot_turn.size() || slot >= wait->stride ||
      coordinate >= wait->coordinate_count ||
      coordinate % wait->stride != slot || turn != coordinate / wait->stride) {
    return accel::PreparedResidencySlidingProjection::Failed;
  }
  if (coordinate == wait->fail_coordinate) {
    return accel::PreparedResidencySlidingProjection::Failed;
  }
  if (wait->reentrant_wake &&
      wait->reentrant_wake_count < wait->reentrant_wake_limit &&
      wait->control != nullptr) {
    const bool woke =
        accel::WakePreparedKernelPipelineSliding(*wait->control).ok;
    wait->reentrant_woke = wait->reentrant_woke || woke;
    wait->reentrant_wake_count += woke ? 1u : 0u;
  }
  if (coordinate == wait->block_coordinate) {
    std::unique_lock lock{wait->gate};
    wait->project_entered = true;
    wait->ready.notify_all();
    wait->ready.wait(lock, [&] { return wait->project_open; });
  }
  if (wait->pending_once && !wait->pending_returned) {
    wait->pending_returned = true;
    return accel::PreparedResidencySlidingProjection::Pending;
  }
  if (wait->pending_remaining != 0u) {
    --wait->pending_remaining;
    return accel::PreparedResidencySlidingProjection::Pending;
  }
  {
    std::lock_guard lock{wait->gate};
    if (turn != wait->slot_turn[slot] ||
        (wait->slot_last[slot] != std::numeric_limits<std::uint64_t>::max() &&
         wait->slot_last[slot] + wait->stride != coordinate)) {
      wait->valid = false;
      return accel::PreparedResidencySlidingProjection::Failed;
    }
    wait->slot_last[slot] = coordinate;
    ++wait->slot_turn[slot];
  }
  selection.locals[0u] = 0u;
  selection.local_count = 1u;
  selection.read_mask = 1u;
  selection.write_mask = 1u;
  selection.descriptor_generation = turn + 1u;
  selection.control_generation = static_cast<std::uint32_t>(coordinate + 2u);
  return accel::PreparedResidencySlidingProjection::Ready;
}

void CompleteSlidingRelease(
    void *const raw,
    accel::PreparedResidencySlidingRelease &&release) noexcept {
  auto *const wait = static_cast<SlidingWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  if (release.terminal.slot < 32u) {
    wait->release_started_mask.fetch_or(
        std::uint32_t{1u} << release.terminal.slot, std::memory_order_release);
  }
  if (release.terminal.coordinate < 64u) {
    wait->release_coordinate_mask.fetch_or(std::uint64_t{1u}
                                               << release.terminal.coordinate,
                                           std::memory_order_release);
  }
  std::unique_lock lock{wait->gate};
  if (release.terminal.coordinate == wait->block_release_coordinate) {
    wait->release_entered = true;
    wait->ready.notify_all();
    wait->ready.wait(lock, [&] { return wait->release_open; });
  }
  wait->valid =
      wait->valid && release.terminal.check.ok &&
      release.terminal.terminal == accel::NativeTerminal::Known &&
      release.terminal.coordinate < wait->coordinate_count &&
      release.terminal.turn == release.terminal.coordinate / wait->stride &&
      release.terminal.slot == release.terminal.coordinate % wait->stride &&
      release.terminal.queue_calls == 1u && release.terminal.dispatched &&
      release.terminal.completed && release.terminal.may_write &&
      release.evidence.check.ok && release.evidence.submitted &&
      release.evidence.control_valid;
  ++wait->releases;
  wait->advanced_past_delay =
      wait->advanced_past_delay ||
      (release.terminal.coordinate > 6u && !wait->delay_opened);
  wait->ready.notify_all();
}

bool CompleteSlidingReturned(void *const raw, const std::uint64_t coordinate,
                             std::uint64_t, std::uint8_t) noexcept {
  auto *const wait = static_cast<SlidingWait *>(raw);
  if (wait == nullptr) {
    return false;
  }
  std::lock_guard lock{wait->gate};
  if (coordinate == wait->block_returned_coordinate && !wait->returned_open &&
      (!wait->returned_opens_on_peer || wait->returned_count == 0u)) {
    wait->returned_entered = true;
    wait->ready.notify_all();
    return false;
  }
  ++wait->returned_count;
  return true;
}

void CompleteSlidingFinal(
    void *const raw, accel::BackendResidencySlidingFinal &&final) noexcept {
  auto *const wait = static_cast<SlidingWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  {
    std::lock_guard lock{wait->gate};
    wait->evidence = std::move(final);
    wait->final = true;
    wait->final_observed.store(true, std::memory_order_release);
  }
  wait->ready.notify_one();
}

SlidingFixture::SlidingFixture() {
  ops.api = rund::AccelApi::Metal;
  ops.seed_prepared_pipeline_generation = FakeSeed;
  ops.submit_prepared_sliding = FakeSubmit;
  ops.prepared_sliding_capability = FakeCapability;
  context_owner = std::make_shared<std::uint8_t>(1u);
  const rund::AccelDevice raw_pick{
      .check = {true, "ok"},
      .api = rund::AccelApi::Metal,
  };
  pick = std::make_shared<accel::PickToken>(raw_pick, ops);
  context = rund::AccelContext{
      .check = {true, "ok"},
      .id = 91u,
      .pick = rund::AccelDevice{.check = {true, "ok"},
                                .api = rund::AccelApi::Metal,
                                .owner = pick},
      .api = rund::AccelApi::Metal,
      .owner = context_owner,
  };
}

[[nodiscard]] bool SlidingFixture::BuildRoles(const std::size_t count) {
  if (count == 0u || count > roles.size()) {
    return false;
  }
  role_count = count;
  for (std::size_t slot = 0u; slot < role_count; ++slot) {
    backends[slot] = std::make_shared<FakeSlidingBackend>();
    states[slot] = std::make_shared<prepared::PipelineState>();
    states[slot]->context = context;
    states[slot]->ops = &ops;
    states[slot]->backend = backends[slot];
    states[slot]->states =
        std::make_unique<std::shared_ptr<prepared::RunState>[]>(1u);
    states[slot]->states[0u] = std::make_shared<prepared::RunState>();
    states[slot]->states[0u]->execution.admission.check = {true, "ok"};
    states[slot]->states[0u]->execution.context_admission.check = {true, "ok"};
    states[slot]->states[0u]->execution.context_admission.context_id =
        context.id;
    states[slot]->states[0u]->execution.context_admission.api = context.api;
    states[slot]->states[0u]->execution.context_admission.owner = context_owner;
    states[slot]->states[0u]->execution.context_admission.pick = pick;
    states[slot]->state_count = 1u;
    states[slot]->size = 1u;
    states[slot]->status.active_step_count = 1u;
    states[slot]->status.declared_step_count = 1u;
    states[slot]->status.declared_steps[0u] = 0u;
    roles[slot] = accel::PreparedResidencySlidingRole{
        .pipeline =
            accel::PreparedKernelPipeline{.owner = states[slot], .ok = true},
        .first_control_generation = static_cast<std::uint32_t>(slot + 2u),
        .control_generation_stride = static_cast<std::uint32_t>(role_count),
        .slot = static_cast<std::uint8_t>(slot),
    };
  }
  return true;
}

[[nodiscard]] bool SlidingFixture::Prepare(const std::size_t count) {
  if (!BuildRoles(count)) {
    return false;
  }
  return accel::PrepareKernelPipelineSliding(
             context,
             std::span<const accel::PreparedResidencySlidingRole>{roles.data(),
                                                                  role_count},
             accel::ResidencySlidingMemory::HostCoherent, control)
      .ok;
}

void SlidingFixture::StartWorkers() {
  for (std::size_t slot = 0u; slot < role_count; ++slot) {
    backends[slot]->worker = std::thread{BackendWorker, backends[slot]};
  }
}

[[nodiscard]] accel::PreparedResidencySlidingRequest
MakeRequest(const SlidingFixture &fixture, SlidingWait &wait,
            const std::uint64_t q, const std::uint64_t plan,
            const std::uint64_t token,
            const std::uint64_t generation) noexcept {
  return accel::PreparedResidencySlidingRequest{
      .plan_identity = plan,
      .token = token,
      .generation = generation,
      .coordinate_count = q,
      .roles = fixture.roles,
      .role_count = fixture.role_count,
      .memory = accel::ResidencySlidingMemory::HostCoherent,
      .project = ProjectSliding,
      .release = CompleteSlidingRelease,
      .returned = CompleteSlidingReturned,
      .final = CompleteSlidingFinal,
      .user = &wait,
  };
}

[[nodiscard]] bool WaitForFinal(SlidingWait &wait) {
  std::unique_lock lock{wait.gate};
  return wait.ready.wait_for(lock, std::chrono::seconds{5},
                             [&] { return wait.final; });
}

[[nodiscard]] bool WaitForReleaseMask(SlidingWait &wait,
                                      const std::uint32_t mask) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while ((wait.release_started_mask.load(std::memory_order_acquire) & mask) !=
         mask) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

} // namespace rund_node_test_pipeline_residency::native_sliding
