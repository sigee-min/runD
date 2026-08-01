#include "claim.hpp"
#include "local.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace rund::compute::detail {

Status acquire_claims(DeviceState &device,
                      const std::span<const BufferClaim> claims,
                      const bool reject_poison) noexcept {
  if (device.claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  std::lock_guard lock{device.claims->gate};
  std::size_t acquired = 0u;
  const auto rollback = [&]() noexcept {
    for (std::size_t index = 0u; index < acquired; ++index) {
      const BufferClaim claim = claims[index];
      if (claim.write) {
        claim.buffer->writer = false;
      } else {
        --claim.buffer->readers;
      }
    }
  };
  for (const BufferClaim claim : claims) {
    Reason failure = Reason::Ok;
    if (claim.buffer == nullptr || claim.buffer->device.get() != &device) {
      failure = Reason::BindingDeviceMismatch;
    } else if (reject_poison && claim.buffer->poisoned) {
      failure = Reason::BufferPoisoned;
    } else if (claim.write ? claim.buffer->writer || claim.buffer->readers != 0u
                           : claim.buffer->writer) {
      failure = Reason::BufferBusy;
    } else if (!claim.write &&
               claim.buffer->readers ==
                   std::numeric_limits<
                       decltype(claim.buffer->readers)>::max()) {
      failure = Reason::BufferBusy;
    }
    if (failure != Reason::Ok) {
      rollback();
      return Status::fail(failure);
    }
    if (claim.write) {
      claim.buffer->writer = true;
    } else {
      ++claim.buffer->readers;
    }
    ++acquired;
  }
  return Status::success();
}

void release_claims(DeviceState &device,
                    const std::span<const BufferClaim> claims) noexcept {
  if (device.claims == nullptr) {
    return;
  }
  std::lock_guard lock{device.claims->gate};
  for (const BufferClaim claim : claims) {
    if (claim.buffer == nullptr) {
      continue;
    }
    if (claim.write) {
      claim.buffer->writer = false;
    } else if (claim.buffer->readers != 0u) {
      --claim.buffer->readers;
    }
  }
}

void publish_claims(DeviceState &device,
                    const std::span<const BufferClaim> claims,
                    const bool succeeded, const bool poison_writes) noexcept {
  if (device.claims == nullptr) {
    return;
  }
  std::lock_guard lock{device.claims->gate};
  for (const BufferClaim claim : claims) {
    if (claim.buffer == nullptr) {
      continue;
    }
    if (claim.write) {
      if (succeeded) {
        ++claim.buffer->generation;
      } else if (poison_writes) {
        // Failure publication is ordered before writer release.
        claim.buffer->poisoned = true;
      }
      claim.buffer->writer = false;
    } else if (claim.buffer->readers != 0u) {
      --claim.buffer->readers;
    }
  }
}

bool buffer_poisoned(const BufferState &buffer) noexcept {
  if (buffer.device == nullptr || buffer.device->claims == nullptr) {
    return true;
  }
  std::lock_guard lock{buffer.device->claims->gate};
  return buffer.poisoned;
}

Status validate_pipeline_resources(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  std::lock_guard lock{state.device->claims->gate};
  for (const PipelineResource &resource : state.resources) {
    if (resource.buffer == nullptr || resource.buffer->device != state.device) {
      return Status::fail(Reason::BindingDeviceMismatch);
    }
    if (resource.buffer->poisoned) {
      return Status::fail(Reason::BufferPoisoned);
    }
  }
  return Status::success();
}

Status acquire_pipeline_claims(PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const std::span<const BufferClaim> claims =
      state.transactional && state.parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  return acquire_claims(*state.device, claims);
}

void publish_pipeline_terminal(PipelineState &state,
                               const PipelineTerminal terminal) noexcept {
  const bool succeeded = terminal.reason == Reason::Ok;
  state.failure = terminal.reason;
  state.failure_step_known = !succeeded && terminal.failure_step_known;
  state.stats.pipeline.verified_step_count =
      succeeded ? state.logical_step_count
                : logical_verified_steps(state, terminal.verified);
  state.stats.pipeline.failed_step_index =
      !succeeded && terminal.failure_step_known &&
              terminal.failed_step < state.steps.size()
          ? logical_step_index(state, terminal.failed_step)
          : PipelineStats::no_failed_step;
  state.stats.pipeline.coalesced_repetition_count =
      succeeded && state.sealed_repetitions != 0u
          ? static_cast<std::uint64_t>(state.sealed_repetitions - 1u)
          : 0u;

  const bool writes_may_have_changed = !succeeded && terminal.writes_possible;
  if (state.device == nullptr || state.device->claims == nullptr) {
    state.failure = Reason::DeviceInvalid;
    state.failure_step_known = false;
    state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
    state.phase = writes_may_have_changed ? PipelinePhase::Poisoned
                                          : PipelinePhase::Ready;
    return;
  }

  if (succeeded) {
    ++state.generation;
    state.stats.publication.generation = state.generation;
    if (state.transactional) {
      ++state.stats.publication.commit_count;
    }
    state.unobserved_outputs = state.outputs.size();
    state.stats.output_hash = 0u;
  }

  const std::span<const BufferClaim> claims =
      state.transactional && state.parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  bool poison_non_state = false;
  if (writes_may_have_changed &&
      (state.transactional ||
       (!state.publications.empty() && terminal.publication_suppressed))) {
    ++state.stats.publication.discard_count;
  }
  if (!succeeded && terminal.reason == Reason::DeviceLost) {
    state.device_lost = true;
    ++state.stats.publication.device_loss_count;
  }

  // Claims use canonical resource order. Pipeline outputs are the write subset
  // of that same order, so generation snapshot, poison, and release need only
  // one resource pass under one Device-gate acquisition.
  std::lock_guard claim_lock{state.device->claims->gate};
  std::size_t output = 0u;
  for (std::size_t resource_index = 0u; resource_index < claims.size();
       ++resource_index) {
    const BufferClaim claim = claims[resource_index];
    if (claim.buffer == nullptr) {
      continue;
    }
    if (claim.write) {
      if (succeeded) {
        ++claim.buffer->generation;
        if (output < state.outputs.size()) {
          PipelineOutputState &published = state.outputs[output++];
          published.generation = claim.buffer->generation;
          published.observed = false;
        }
      } else if (writes_may_have_changed && !claim.transactional_state &&
                 !(claim.gated_publish && terminal.publication_suppressed) &&
                 resource_index < state.resources.size() &&
                 (!terminal.failure_step_known ||
                  state.resources[resource_index].first_write <=
                      terminal.failed_step)) {
        // Poison is visible before this writer claim is released.
        claim.buffer->poisoned = true;
        poison_non_state = true;
      }
      claim.buffer->writer = false;
    } else if (claim.buffer->readers != 0u) {
      --claim.buffer->readers;
    }
  }
  if (succeeded && state.transactional) {
    state.parity ^= 1u;
  }
  state.phase = state.control_poisoned           ? PipelinePhase::Poisoned
                : succeeded || !poison_non_state ? PipelinePhase::Ready
                                                 : PipelinePhase::Poisoned;
}

} // namespace rund::compute::detail
