#include "local.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency::native_sliding {

[[nodiscard]] bool AdmissionRejectNoAllocationCase() {
  SlidingFixture fixture{};
  if (!fixture.BuildRoles(1u)) {
    return false;
  }
  fixture.ops.prepared_sliding_capability = RejectedCapability;
  node_compute_allocation::Start();
  const rund::AccelCheck admitted = accel::PrepareKernelPipelineSliding(
      fixture.context,
      std::span<const accel::PreparedResidencySlidingRole>{fixture.roles.data(),
                                                           fixture.role_count},
      accel::ResidencySlidingMemory::HostCoherent, fixture.control);
  node_compute_allocation::Stop();
  return !admitted.ok && fixture.control.state == nullptr &&
         node_compute_allocation::Count() == 0u;
}

[[nodiscard]] bool SlidingCase(const std::uint64_t q) {
  SlidingFixture fixture{};
  if (!fixture.Prepare()) {
    return false;
  }
  const std::uint64_t retained = fixture.control.capability.retained_bytes;
  const std::uint64_t transient = fixture.control.capability.transient_bytes;

  SlidingWait wait{};
  wait.coordinate_count = q;
  wait.stride = fixture.role_count;
  wait.control = &fixture.control;
  wait.reentrant_wake = true;
  wait.pending_once = true;
  fixture.StartWorkers();
  // Delay the first reuse of slot 2. Other slots continue independently; this
  // proves that the transport has no rigid W4 chunk barrier.
  fixture.backends[1u]->delay_generation = 2u;
  fixture.backends[2u]->delay_generation = 7u;
  const accel::PreparedResidencySlidingRequest request =
      MakeRequest(fixture, wait, q);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  {
    std::lock_guard lock{fixture.backends[1u]->gate};
    fixture.backends[1u]->released = true;
  }
  fixture.backends[1u]->ready.notify_all();
  if (q > 7u) {
    {
      std::unique_lock lock{fixture.backends[2u]->gate};
      if (!fixture.backends[2u]->ready.wait_for(
              lock, std::chrono::seconds{5}, [&] {
                return fixture.backends[2u]->submitted &&
                       fixture.backends[2u]->seeded == 7u;
              })) {
        return false;
      }
    }
    {
      std::unique_lock lock{wait.gate};
      wait.ready.wait(lock, [&] { return wait.advanced_past_delay; });
      wait.delay_opened = true;
    }
    {
      std::lock_guard lock{fixture.backends[2u]->gate};
      fixture.backends[2u]->delay_open = true;
      fixture.backends[2u]->released = true;
    }
    fixture.backends[2u]->ready.notify_one();
  }
  {
    std::unique_lock lock{wait.gate};
    if (!wait.ready.wait_for(lock, std::chrono::seconds{5},
                             [&] { return wait.final; })) {
      std::fprintf(
          stderr,
          "native sliding q=%llu final timeout releases=%llu "
          "release-mask=%x coordinates=%llx wake=%zu pending=%zu "
          "advanced=%u submissions=%llu/%llu/%llu/%llu\n",
          static_cast<unsigned long long>(q),
          static_cast<unsigned long long>(wait.releases),
          wait.release_started_mask.load(std::memory_order_acquire),
          static_cast<unsigned long long>(
              wait.release_coordinate_mask.load(std::memory_order_acquire)),
          wait.reentrant_wake_count, wait.pending_remaining,
          static_cast<unsigned>(wait.advanced_past_delay),
          static_cast<unsigned long long>(
              fixture.backends[0u]->submission_count),
          static_cast<unsigned long long>(
              fixture.backends[1u]->submission_count),
          static_cast<unsigned long long>(
              fixture.backends[2u]->submission_count),
          static_cast<unsigned long long>(
              fixture.backends[3u]->submission_count));
      return false;
    }
  }
  const bool valid = wait.valid && wait.reentrant_woke &&
                     (q <= 7u || wait.advanced_past_delay) &&
                     wait.releases == q && wait.evidence.check.ok &&
                     wait.evidence.terminal == accel::NativeTerminal::Known &&
                     wait.evidence.coordinate_count == q &&
                     wait.evidence.accepted_coordinates == q &&
                     wait.evidence.released_coordinates == q &&
                     wait.evidence.queue_calls == q &&
                     wait.evidence.native_inflight_peak >= 2u &&
                     wait.evidence.native_inflight_peak <= fixture.role_count &&
                     wait.evidence.retained_bytes == retained &&
                     wait.evidence.transient_bytes == transient &&
                     wait.evidence.completed_ns != 0u;
  if (!valid) {
    std::fprintf(
        stderr,
        "native sliding q=%llu valid=%u releases=%llu check=%u "
        "accepted=%llu retired=%llu calls=%llu peak=%llu bytes=%llu/%llu "
        "expected=%llu/%llu\n",
        static_cast<unsigned long long>(q), static_cast<unsigned>(wait.valid),
        static_cast<unsigned long long>(wait.releases),
        static_cast<unsigned>(wait.evidence.check.ok),
        static_cast<unsigned long long>(wait.evidence.accepted_coordinates),
        static_cast<unsigned long long>(wait.evidence.released_coordinates),
        static_cast<unsigned long long>(wait.evidence.queue_calls),
        static_cast<unsigned long long>(wait.evidence.native_inflight_peak),
        static_cast<unsigned long long>(wait.evidence.retained_bytes),
        static_cast<unsigned long long>(wait.evidence.transient_bytes),
        static_cast<unsigned long long>(retained),
        static_cast<unsigned long long>(transient));
  }
  return valid;
}

[[nodiscard]] bool InvalidMaskCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  const auto original_project = ProjectSliding;
  const auto invalid_project =
      +[](void *const raw, const std::uint64_t c, const std::uint64_t turn,
          const std::uint8_t slot,
          accel::PreparedResidencySlidingSelection &selection) noexcept {
        const auto result = ProjectSliding(raw, c, turn, slot, selection);
        selection.write_mask = 2u;
        return result;
      };
  auto request = MakeRequest(fixture, wait, 1u);
  request.project = invalid_project;
  if (original_project == nullptr ||
      !accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok ||
      !WaitForFinal(wait)) {
    return false;
  }
  return wait.evidence.terminal == accel::NativeTerminal::Known &&
         !wait.evidence.check.ok && wait.evidence.accepted_coordinates == 0u &&
         fixture.backends[0u]->submission_count == 0u;
}

[[nodiscard]] bool StrideCase(const std::size_t roles) {
  SlidingFixture fixture{};
  if (!fixture.Prepare(roles)) {
    return false;
  }
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = roles * 2u + 1u;
  wait.stride = roles;
  const auto request = MakeRequest(fixture, wait, wait.coordinate_count);
  return accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                    fixture.control)
             .ok &&
         WaitForFinal(wait) && wait.valid && wait.evidence.check.ok &&
         wait.evidence.accepted_coordinates == wait.coordinate_count &&
         wait.evidence.released_coordinates == wait.coordinate_count;
}

} // namespace rund_node_test_pipeline_residency::native_sliding
