#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {
namespace {

[[nodiscard]] bool failure_precedes(const std::uint64_t coordinate,
                                    const rund::AccelCheck failure,
                                    const State &state) noexcept {
  if (coordinate != state.first_failure) {
    return coordinate < state.first_failure;
  }
  const char *const incoming = failure.reason == nullptr ? "" : failure.reason;
  const char *const current =
      state.failure.reason == nullptr ? "" : state.failure.reason;
  return std::strcmp(incoming, current) < 0;
}

} // namespace

bool valid_memory(const BackendResidencySlidingCapability &capability,
                  const ResidencySlidingMemory memory) noexcept {
  if (!capability.check.ok || !capability.callbacks_async ||
      !capability.descriptor_release_acquire || capability.max_slots == 0u ||
      capability.memory != memory) {
    return false;
  }
  switch (memory) {
  case ResidencySlidingMemory::HostCoherent:
    return !capability.integrated_copy && !capability.distinct_transfer_queue;
  case ResidencySlidingMemory::NoncoherentIntegratedCopy:
    return capability.flush_before_frontier && capability.integrated_copy;
  case ResidencySlidingMemory::NoncoherentSplitTransfer:
    return capability.flush_before_frontier &&
           capability.distinct_transfer_queue;
  }
  return false;
}

bool same_capability(const BackendResidencySlidingCapability &left,
                     const BackendResidencySlidingCapability &right) noexcept {
  return left.memory == right.memory && left.max_slots == right.max_slots &&
         left.callbacks_async == right.callbacks_async &&
         left.descriptor_release_acquire == right.descriptor_release_acquire &&
         left.flush_before_frontier == right.flush_before_frontier &&
         left.integrated_copy == right.integrated_copy &&
         left.distinct_transfer_queue == right.distinct_transfer_queue;
}

bool valid_selection(
    const PreparedResidencySlidingSelection &selection) noexcept {
  if (selection.local_count == 0u ||
      selection.local_count > ResidencyWindowLocalCapacity ||
      selection.control_generation == 0u ||
      selection.descriptor_generation == 0u ||
      (selection.read_mask == 0u && selection.write_mask == 0u)) {
    return false;
  }
  const std::uint64_t active_mask =
      selection.local_count == std::numeric_limits<std::uint64_t>::digits
          ? std::numeric_limits<std::uint64_t>::max()
          : (std::uint64_t{1u} << selection.local_count) - 1u;
  if ((selection.read_mask & ~active_mask) != 0u ||
      (selection.write_mask & ~active_mask) != 0u) {
    return false;
  }
  for (std::size_t local = 0u; local < selection.local_count; ++local) {
    for (std::size_t prior = 0u; prior < local; ++prior) {
      if (selection.locals[local] == selection.locals[prior]) {
        return false;
      }
    }
  }
  return true;
}

bool same_roles(const State &state,
                const PreparedResidencySlidingRequest &request) noexcept {
  if (request.role_count != state.role_count) {
    return false;
  }
  for (std::size_t slot = 0u; slot < state.role_count; ++slot) {
    if (request.roles[slot].slot != slot ||
        request.roles[slot].first_control_generation !=
            state.roles[slot].first_control_generation ||
        request.roles[slot].control_generation_stride !=
            state.roles[slot].control_generation_stride ||
        request.roles[slot].pipeline.owner !=
            state.roles[slot].pipeline.owner ||
        request.roles[slot].pipeline.ok != state.roles[slot].pipeline.ok) {
      return false;
    }
  }
  return true;
}

void release_claims_known(State &state) noexcept {
  std::array<std::unique_lock<std::mutex>, ResidencySlidingCapacity> claims{};
  for (std::size_t slot = 0u; slot < state.role_count; ++slot) {
    claims[slot] =
        std::unique_lock<std::mutex>{state.pipelines[slot]->submission.mutex};
  }
  for (std::size_t slot = 0u; slot < state.role_count; ++slot) {
    prepared::PipelineSubmission &submission =
        state.pipelines[slot]->submission;
    if (submission.sliding == &state) {
      submission.sliding = nullptr;
    }
  }
}

void record_failure(State &state, const std::uint64_t coordinate,
                    const rund::AccelCheck failure) noexcept {
  state.failed = true;
  if (failure_precedes(coordinate, failure, state)) {
    state.first_failure = coordinate;
    state.failure = failure;
  }
}

} // namespace rund::node::accel::detail::prepared::sliding
