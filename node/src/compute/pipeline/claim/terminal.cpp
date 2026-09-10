#include "internal.hpp"

#include "../local.hpp"
#include "../sample.hpp"

#include <limits>
#include <mutex>

namespace rund::compute::detail::claim_detail {

PrivateTerminalCheck
private_terminal_check(const PipelineState &state,
                       const PipelineTerminal terminal,
                       const bool private_authority_proven) noexcept {
  const bool authority_valid = has_private_residency_authority(state);
  const bool proof_valid = !private_authority_proven || authority_valid;
  const bool registry = authority_valid && proof_valid;
  const bool identity = !terminal.defer_generation ||
                        (registry &&
                         terminal.publication_generation !=
                             PipelineTerminal::no_publication_generation &&
                         terminal.publication_parity <= 1u);
  const std::uint64_t publication_generation =
      terminal.defer_generation ? terminal.publication_generation
                                : state.attempt.generation;
  const std::uint8_t publication_parity = terminal.defer_generation
                                              ? terminal.publication_parity
                                              : state.attempt.parity;
  const bool publication =
      state.publication != nullptr && state.publication->attempt_active &&
      state.publication->generation == publication_generation &&
      state.publication->parity == publication_parity;
  const bool phase = state.phase == PipelinePhase::Running;
  return PrivateTerminalCheck{
      .registry = registry,
      .identity = identity,
      .publication = publication,
      .phase = phase,
      .valid = registry && identity && publication && phase,
  };
}

} // namespace rund::compute::detail::claim_detail

namespace rund::compute::detail {

void publish_pipeline_terminal_locked(
    PipelineState &state, const PipelineTerminal terminal,
    const PipelineClaimAuthority authority, const bool private_authority_proven,
    const bool shared_claim_gate_proven) noexcept {
  const bool deferred = terminal.defer_generation;
  const claim_detail::PrivateTerminalCheck private_check =
      authority == PipelineClaimAuthority::PrivateResidency
          ? claim_detail::private_terminal_check(state, terminal,
                                                 private_authority_proven)
          : claim_detail::PrivateTerminalCheck{};
  const bool private_registry = private_check.registry;
  const bool publication_matches =
      private_check.publication ||
      (authority != PipelineClaimAuthority::PrivateResidency &&
       state.publication->attempt_active &&
       state.publication->generation == state.attempt.generation &&
       state.publication->parity == state.attempt.parity);
  const bool private_authority_invalid =
      (authority == PipelineClaimAuthority::PrivateResidency &&
       !private_check.valid) ||
      (authority != PipelineClaimAuthority::PrivateResidency && deferred);
  const bool succeeded = terminal.reason == Reason::Ok && publication_matches &&
                         !private_authority_invalid;
  const Reason reason =
      private_authority_invalid
          ? Reason::PipelineInvalid
          : (publication_matches ? terminal.reason : Reason::CompletionInvalid);
  if (state.samples != PipelineState::SampleState::Inactive) {
    const bool clean = succeeded &&
                       state.samples == PipelineState::SampleState::Clean &&
                       pipeline_sample_is_clean(state.stats);
    const auto increment = [](std::uint32_t &value) noexcept {
      if (value != std::numeric_limits<std::uint32_t>::max()) {
        ++value;
      }
    };
    increment(state.stats.pipeline.sampled_runs);
    if (clean) {
      increment(state.stats.pipeline.clean_runs);
    } else {
      state.samples = PipelineState::SampleState::Dirty;
    }
  }
  state.failure = reason;
  state.attempt.failure_step_known = !succeeded && terminal.failure_step_known;
  state.stats.pipeline.verified_step_count =
      succeeded ? (terminal.issued_steps == 0u
                       ? state.logical_step_count
                       : logical_verified_steps(state, terminal.issued_steps))
                : (private_registry && terminal.issued_steps != 0u
                       ? terminal.verified
                       : logical_verified_steps(state, terminal.verified));
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
    if (deferred) {
      // A deferred physical terminal advances only the private control
      // identity. The run-level cursor owns canonical publication and
      // observation updates after every aggregate terminal succeeds.
      state.native_generation = state.attempt.generation + 1u;
      state.native_parity = state.attempt.parity;
    } else {
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
  }

  const std::span<const BufferClaim> claims =
      state.transactional && state.attempt.parity != 0u
          ? std::span<const BufferClaim>{state.alternate_claims}
          : std::span<const BufferClaim>{state.claims};
  bool poison_non_state = false;
  if (!deferred && writes_may_have_changed &&
      (state.transactional ||
       (!state.publications.empty() && terminal.publication_suppressed))) {
    ++state.stats.publication.discard_count;
  }
  if (!succeeded && reason == Reason::DeviceLost) {
    state.publication->device_lost = true;
    ++state.stats.publication.device_loss_count;
  }

  const auto publish_resources = [&]() noexcept {
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
          claim.buffer->poisoned = true;
          poison_non_state = true;
        }
        claim.buffer->writer = false;
      } else if (claim.buffer->readers != 0u) {
        --claim.buffer->readers;
      }
    }
  };
  if (private_registry) {
    // Registry Authority owns validity, dirty state, and the exact issued
    // prefix for these private physical arenas. Publishing Buffer generation
    // here would widen an n-frame lease to the K-frame owner and create a
    // second memory-validity authority. No shared claims were acquired for
    // this mode, so there is also nothing to release.
  } else if (shared_claim_gate_proven) {
    publish_resources();
  } else {
    std::lock_guard claim_lock{state.device->claims->gate};
    publish_resources();
  }
  if (succeeded && !deferred && state.transactional) {
    state.publication->parity ^= 1u;
  }
  if (succeeded && !deferred) {
    state.native_parity = state.publication->parity;
    state.observation_generation = state.publication->generation;
    state.observation_payload_epoch = state.publication->payload_epoch;
    state.observation_parity = state.publication->parity;
    state.observation_identity_valid = true;
  }
  state.publication->attempt_active = false;
  state.phase = state.control_poisoned || private_authority_invalid
                    ? PipelinePhase::Poisoned
                : succeeded || !poison_non_state ? PipelinePhase::Ready
                                                 : PipelinePhase::Poisoned;
}

bool shared_pipeline_terminal_ready(const PipelineState &state) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr ||
      state.publication == nullptr || state.phase != PipelinePhase::Running) {
    return false;
  }
  std::lock_guard publication_lock{state.publication->gate};
  return state.publication->attempt_active &&
         state.publication->generation == state.attempt.generation &&
         state.publication->parity == state.attempt.parity;
}

void publish_shared_pipeline_terminal_transaction(
    PipelineState &state, const PipelineTerminal terminal) noexcept {
  std::lock_guard publication_lock{state.publication->gate};
  std::lock_guard claim_lock{state.device->claims->gate};
  publish_pipeline_terminal_locked(state, terminal,
                                   PipelineClaimAuthority::Shared, false, true);
}

void publish_pipeline_terminal(
    PipelineState &state, const PipelineTerminal terminal,
    const PipelineClaimAuthority authority) noexcept {
  if (state.device == nullptr || state.device->claims == nullptr ||
      state.publication == nullptr) {
    if (state.publication != nullptr) {
      std::lock_guard publication_lock{state.publication->gate};
      if (state.publication->attempt_active &&
          state.publication->generation == state.attempt.generation &&
          state.publication->parity == state.attempt.parity) {
        state.publication->attempt_active = false;
      }
    }
    state.failure = Reason::DeviceInvalid;
    state.attempt.failure_step_known = false;
    state.stats.pipeline.failed_step_index = PipelineStats::no_failed_step;
    state.phase = terminal.writes_possible ? PipelinePhase::Poisoned
                                           : PipelinePhase::Ready;
    return;
  }

  // Pipeline callers already own state.gate. Publication is the second lock
  // in the global order; the Device claim gate below is always third.
  std::lock_guard publication_lock{state.publication->gate};
  publish_pipeline_terminal_locked(state, terminal, authority);
}

} // namespace rund::compute::detail
