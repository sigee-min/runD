#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../backend.hpp"
#include "../../status.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"
#include "../transfer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {

Result<std::shared_ptr<PipelinePublicationState>>
latest_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept {
  if (!valid_pipeline(state)) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::ProfileBusy);
  }
  if (state->publication == nullptr) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  if (state->publication->device_lost) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::DeviceLost);
  }
  if (!state->transactional || state->publication->state_pairs.empty() ||
      state->publication->device == nullptr ||
      !state->publication->fingerprint) {
    return Result<std::shared_ptr<PipelinePublicationState>>::fail(
        Reason::PipelineInvalid);
  }
  ::rund::detail::counter::Accumulate(
      state->checkpoint_stats.device_state_acquire_count, 1u);
  return Result<std::shared_ptr<PipelinePublicationState>>::success(
      state->publication);
}

Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<PipelinePublicationState> &source) noexcept {
  if (!valid_pipeline(state) || source == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock pipeline_lock{state->gate, std::try_to_lock};
  if (!pipeline_lock.owns_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->samples != PipelineState::SampleState::Inactive) {
    return Status::fail(Reason::ProfileBusy);
  }
  const std::shared_ptr<PipelinePublicationState> target = state->publication;
  if (target == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }

  std::unique_lock<std::mutex> target_lock{target->gate, std::defer_lock};
  std::unique_lock<std::mutex> source_lock{source->gate, std::defer_lock};
  if (target == source) {
    if (!target_lock.try_lock()) {
      return Status::fail(Reason::PipelineBusy);
    }
  } else if (std::less<const PipelinePublicationState *>{}(target.get(),
                                                           source.get())) {
    if (!target_lock.try_lock() || !source_lock.try_lock()) {
      return Status::fail(Reason::PipelineBusy);
    }
  } else if (!source_lock.try_lock() || !target_lock.try_lock()) {
    return Status::fail(Reason::PipelineBusy);
  }

  if (target->device_lost || source->device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  if (target->attempt_active || source->attempt_active ||
      state->phase == PipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (!state->transactional || target->state_pairs.empty() ||
      source->state_pairs.empty() || state->phase != PipelinePhase::Ready) {
    return Status::fail(state->phase == PipelinePhase::Poisoned
                            ? Reason::PipelinePoisoned
                            : Reason::PipelineInvalid);
  }
  if (source->generation > PipelineGenerationCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (target->device == nullptr || source->device == nullptr ||
      target->device != state->device || source->device != state->device) {
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  if (target->fingerprint != source->fingerprint ||
      target->state_pairs.size() != source->state_pairs.size() ||
      target->state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  bool ordered_owners_equal = true;
  for (std::size_t index = 0u; index < target->state_pairs.size(); ++index) {
    const PipelineStatePair &destination = target->state_pairs[index];
    const PipelineStatePair &published = source->state_pairs[index];
    if (destination.type != published.type ||
        destination.format != published.format ||
        destination.count != published.count ||
        destination.bytes != published.bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    ordered_owners_equal = ordered_owners_equal &&
                           destination.first == published.first &&
                           destination.second == published.second;
  }

  if (ordered_owners_equal) {
    if (target != source && !state->preparing) {
      // Replacing a live authority would strand any handle already issued by
      // the destination. Nothing has been mutated yet, so reject atomically
      // and preserve the destination's current publication and phase.
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status seeded =
        seed_pipeline_generations(*state, source->generation, source->parity);
    if (!seeded) {
      if (seeded.reason() == Reason::DeviceLost) {
        target->device_lost = true;
        source->device_lost = true;
      }
      state->failure = seeded.reason();
      state->phase = PipelinePhase::Poisoned;
      return seeded;
    }
    if (target != source) {
      state->publication = source;
    }
    state->failure = Reason::Ok;
    state->control_poisoned = false;
    state->stats.publication.generation = source->generation;
    ::rund::detail::counter::Accumulate(
        state->checkpoint_stats.device_state_rebase_count, 1u);
    return Status::success();
  }

  if (target->payload_epoch == std::numeric_limits<std::uint64_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }

  // The copy path is deliberately all-disjoint. Any partial alias (including
  // reversed pair orientation) would require a cycle-breaking temporary and
  // would leave two independent selectors over shared owners.
  for (const PipelineStatePair &destination : target->state_pairs) {
    for (const PipelineStatePair &published : source->state_pairs) {
      if (destination.first == published.first ||
          destination.first == published.second ||
          destination.second == published.first ||
          destination.second == published.second) {
        return Status::fail(Reason::BindingDuplicate);
      }
    }
  }

  const std::size_t pair_count = target->state_pairs.size();
  std::array<BufferClaim, PipelineLeafCapacity * 2u> claim_storage{};
  std::array<CopyRequest, PipelineLeafCapacity> copy_storage{};
  std::size_t copy_count = 0u;
  std::size_t copied = 0u;
  for (std::size_t index = 0u; index < pair_count; ++index) {
    const PipelineStatePair &published = source->state_pairs[index];
    const PipelineStatePair &destination = target->state_pairs[index];
    BufferState *const source_buffer =
        (source->parity == 0u ? published.first : published.second).get();
    BufferState *const target_buffer = destination.first.get();
    claim_storage[index] = BufferClaim{.buffer = source_buffer, .write = false};
    claim_storage[pair_count + index] =
        BufferClaim{.buffer = target_buffer, .write = true};
    if (published.bytes != 0u) {
      copy_storage[copy_count++] = CopyRequest{
          .source = source_buffer,
          .target = target_buffer,
          .bytes = published.bytes,
          .source_offset = 0u,
          .target_offset = 0u,
      };
      copied = ::rund::detail::counter::SaturatingAdd(copied, published.bytes);
    }
  }
  const std::span<const BufferClaim> claims{claim_storage.data(),
                                            pair_count * 2u};
  const Status claimed = acquire_claims(*state->device, claims);
  if (!claimed) {
    return claimed;
  }
  ClaimGuard guard{*state->device, claims};
  Status restored = Status::success();
  std::uint64_t copy_commands = 0u;
  if (state->device->backend == Backend::Cpu) {
    for (std::size_t index = 0u; index < copy_count; ++index) {
      const CopyRequest request = copy_storage[index];
      const CpuBufferState *const source_cpu = cpu_buffer(*request.source);
      CpuBufferState *const target_cpu = cpu_buffer(*request.target);
      if (source_cpu == nullptr || target_cpu == nullptr ||
          source_cpu->data == nullptr || target_cpu->data == nullptr ||
          request.source_offset > source_cpu->bytes ||
          request.bytes > source_cpu->bytes - request.source_offset ||
          request.target_offset > target_cpu->bytes ||
          request.bytes > target_cpu->bytes - request.target_offset) {
        restored = Status::fail(Reason::TransferInvalid);
        break;
      }
      std::memcpy(target_cpu->data.get() + request.target_offset,
                  source_cpu->data.get() + request.source_offset,
                  request.bytes);
    }
  } else if (copy_count != 0u) {
    if (state->device->ops == nullptr ||
        state->device->ops->copy_batch == nullptr) {
      restored = Status::fail(Reason::TransferInvalid);
    } else {
      const CopyResult result = state->device->ops->copy_batch(
          *state->device,
          std::span<const CopyRequest>{copy_storage.data(), copy_count},
          node::accel::detail::TransferAuthority::Shared);
      restored = result.status;
      copy_commands = result.command_submits;
      ::rund::detail::counter::Accumulate(
          state->stats.transfer_submissions.device_to_device,
          result.command_submits);
    }
  }
  if (restored) {
    restored = seed_pipeline_generations(*state, source->generation, 0u);
  }
  if (!restored) {
    if (restored.reason() == Reason::DeviceLost) {
      target->device_lost = true;
      source->device_lost = true;
    }
    publish_claims(*state->device, claims, false, true);
    guard.dismiss();
    state->failure = restored.reason();
    state->phase = PipelinePhase::Poisoned;
    return restored;
  }
  {
    std::lock_guard claim_lock{state->device->claims->gate};
    for (const BufferClaim claim : claims) {
      if (claim.write) {
        claim.buffer->poisoned = false;
        ++claim.buffer->generation;
        claim.buffer->writer = false;
      } else if (claim.buffer->readers != 0u) {
        --claim.buffer->readers;
      }
    }
  }
  guard.dismiss();
  target->parity = 0u;
  target->generation = source->generation;
  ++target->payload_epoch;
  target->device_lost = false;
  close_pipeline_observation_epoch(*state);
  state->failure = Reason::Ok;
  state->control_poisoned = false;
  state->stats.publication.generation = target->generation;
  state->checkpoint_stats.device_state_copy_byte_count =
      ::rund::detail::counter::SaturatingAdd(
          state->checkpoint_stats.device_state_copy_byte_count, copied);
  state->checkpoint_stats.device_state_copy_command_count =
      ::rund::detail::counter::SaturatingAdd(
          state->checkpoint_stats.device_state_copy_command_count,
          copy_commands);
  return Status::success();
}

bool latest_state_valid(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return false;
  }
  std::lock_guard lock{publication->gate};
  return publication->device != nullptr && !publication->state_pairs.empty() &&
         static_cast<bool>(publication->fingerprint) &&
         !publication->device_lost;
}

std::uint64_t latest_state_generation(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return 0u;
  }
  std::lock_guard lock{publication->gate};
  return publication->generation;
}

graph::Fingerprint latest_state_fingerprint(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept {
  if (publication == nullptr) {
    return {};
  }
  std::lock_guard lock{publication->gate};
  return publication->fingerprint;
}

} // namespace rund::compute::detail
