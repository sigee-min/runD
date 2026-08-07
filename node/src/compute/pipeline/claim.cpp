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

Status
validate_pipeline_resource_device(const PipelineState &state,
                                  const PipelineResource &resource) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  return resource.buffer == nullptr || resource.buffer->device != state.device
             ? Status::fail(Reason::BindingDeviceMismatch)
             : Status::success();
}

Status validate_pipeline_resources(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  std::lock_guard lock{state.device->claims->gate};
  for (const PipelineResource &resource : state.resources) {
    const Status device = validate_pipeline_resource_device(state, resource);
    if (!device) {
      return device;
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
      state.transactional && state.attempt_parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  return acquire_claims(*state.device, claims);
}

void close_pipeline_observation_epoch(PipelineState &state) noexcept {
  state.unobserved_outputs = 0u;
  state.observation_identity_valid = false;
  state.stats.output_hash = 0u;
  for (PipelineOutputState &output : state.outputs) {
    output.observed = false;
    output.hash = 0u;
  }
}

void synchronize_pipeline_observation_epoch(
    PipelineState &state,
    const PipelinePublicationState &publication) noexcept {
  if (state.observation_identity_valid &&
      (state.observation_generation != publication.generation ||
       state.observation_parity != publication.parity ||
       state.observation_payload_epoch != publication.payload_epoch)) {
    close_pipeline_observation_epoch(state);
  }
}

void publish_pipeline_terminal(PipelineState &state,
                               const PipelineTerminal terminal) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr ||
      state.publication == nullptr) {
    if (state.publication != nullptr) {
      std::lock_guard publication_lock{state.publication->gate};
      if (state.publication->attempt_active &&
          state.publication->generation == state.attempt_generation &&
          state.publication->parity == state.attempt_parity) {
        state.publication->attempt_active = false;
      }
    }
    state.failure = Reason::DeviceInvalid;
    state.failure_step_known = false;
    state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
    state.phase = terminal.writes_possible ? PipelinePhase::Poisoned
                                           : PipelinePhase::Ready;
    return;
  }

  // Pipeline callers already own state.gate. Publication is the second lock
  // in the global order; the Device claim gate below is always third.
  std::lock_guard publication_lock{state.publication->gate};
  const bool publication_matches =
      state.publication->attempt_active &&
      state.publication->generation == state.attempt_generation &&
      state.publication->parity == state.attempt_parity;
  const bool succeeded = terminal.reason == Reason::Ok && publication_matches;
  const Reason reason =
      publication_matches ? terminal.reason : Reason::CompletionInvalid;
  state.failure = reason;
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

  if (succeeded) {
    ++state.publication->generation;
    ++state.publication->payload_epoch;
    state.native_generation = state.publication->generation;
    state.stats.publication.generation = state.publication->generation;
    if (state.transactional) {
      ++state.stats.publication.commit_count;
    }
    state.unobserved_outputs = state.outputs.size();
    state.stats.output_hash = 0u;
  }

  const std::span<const BufferClaim> claims =
      state.transactional && state.attempt_parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  bool poison_non_state = false;
  if (writes_may_have_changed &&
      (state.transactional ||
       (!state.publications.empty() && terminal.publication_suppressed))) {
    ++state.stats.publication.discard_count;
  }
  if (!succeeded && reason == Reason::DeviceLost) {
    state.publication->device_lost = true;
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
    state.publication->parity ^= 1u;
  }
  if (succeeded) {
    state.native_parity = state.publication->parity;
    state.observation_generation = state.publication->generation;
    state.observation_payload_epoch = state.publication->payload_epoch;
    state.observation_parity = state.publication->parity;
    state.observation_identity_valid = true;
  }
  state.publication->attempt_active = false;
  state.phase = state.control_poisoned           ? PipelinePhase::Poisoned
                : succeeded || !poison_non_state ? PipelinePhase::Ready
                                                 : PipelinePhase::Poisoned;
}

} // namespace rund::compute::detail
