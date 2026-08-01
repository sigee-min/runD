#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../hash/fnv.hpp"
#include "../backend.hpp"
#include "claim.hpp"
#include "state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

namespace rund::compute::detail {
namespace {

void mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  ::rund::node::hash_detail::MixU64(hash, value);
}

} // namespace

Status read_pipeline_raw(const std::shared_ptr<PipelineState> &state,
                         const std::shared_ptr<BufferState> &buffer,
                         const Type type, const FixedFormat format,
                         void *const data, const std::size_t bytes,
                         const std::size_t count) noexcept {
  if (!valid_pipeline(state) || buffer == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard pipeline_lock{state->gate};
  if (state->publication == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard publication_lock{state->publication->gate};
  if (state->publication->device_lost) {
    return Status::fail(Reason::DeviceLost);
  }
  synchronize_pipeline_observation_epoch(*state, *state->publication);
  const bool running = state->phase == PipelinePhase::Running;
  const bool poisoned = state->phase == PipelinePhase::Poisoned;
  if (!running && !poisoned && state->publication->generation == 0u &&
      !state->transactional) {
    return Status::fail(Reason::ResidentNotRun);
  }
  if (!running && !poisoned &&
      (state->outputs.size() != state->output_lookup.size() ||
       state->unobserved_outputs > state->outputs.size())) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const BufferState *requested = buffer.get();
  std::shared_ptr<BufferState> observed_buffer = buffer;
  bool transactional_state_read = false;
  if (state->transactional) {
    for (const PipelineStatePair &pair : state->publication->state_pairs) {
      if (requested == pair.first.get() || requested == pair.second.get()) {
        requested = pair.second.get();
        observed_buffer =
            state->publication->parity == 0u ? pair.first : pair.second;
        transactional_state_read = true;
        break;
      }
    }
  }
  const auto found = std::lower_bound(
      state->output_lookup.begin(), state->output_lookup.end(), requested,
      [&](const std::uint32_t output, const BufferState *const target) {
        if (output >= state->outputs.size() ||
            state->outputs[output].resource >= state->resources.size()) {
          return false;
        }
        return std::less<const BufferState *>{}(
            state->resources[state->outputs[output].resource].buffer.get(),
            target);
      });
  if (found == state->output_lookup.end() || *found >= state->outputs.size()) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  const std::size_t output_index = *found;
  PipelineOutputState &output = state->outputs[output_index];
  if (output.resource >= state->resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t ordinal = output.resource;
  const PipelineResource &shape = state->resources[ordinal];
  if (shape.buffer.get() != requested) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  if (shape.output != output_index) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (shape.type != type || shape.count != count || shape.bytes != bytes ||
      (data == nullptr && bytes != 0u) ||
      ((type == Type::FixedLane32 || type == Type::FixedLane64) &&
       (shape.format.integer_bits != format.integer_bits ||
        shape.format.fraction_bits != format.fraction_bits))) {
    return Status::fail(Reason::ShapeMismatch);
  }
  const BufferClaim claim{observed_buffer.get(), false};
  const Status claimed = acquire_claims(*state->device, {&claim, 1u});
  if (!claimed) {
    return claimed;
  }
  ClaimGuard claim_guard{*state->device, {&claim, 1u}};
  if (poisoned && !transactional_state_read) {
    // Declared writes are poisoned before the execution claims are released,
    // so the claim boundary normally returns BufferPoisoned above.
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (running) {
    // A correctly admitted execution still owns its write claim, so this path
    // normally returns BufferBusy above.  Preserve PipelineBusy as the
    // defensive state-machine result if the claim table is inconsistent.
    return Status::fail(Reason::PipelineBusy);
  }
  if (!transactional_state_read &&
      observed_buffer->generation != output.generation) {
    return Status::fail(Reason::ReadBufferMismatch);
  }
  Status result = Status::success();
  std::uint64_t leaf_hash = ::rund::node::hash_detail::ZeroHash(0u);
  if (bytes != 0u && state->device->backend == Backend::Cpu) {
    const CpuBufferState *const source = cpu_buffer(*observed_buffer);
    if (source == nullptr || source->bytes < bytes) {
      result = Status::fail(Reason::TransferInvalid);
    } else {
      leaf_hash =
          ::rund::node::hash_detail::CopyHash(source->data.get(), data, bytes);
    }
  } else if (bytes != 0u) {
    if (state->device->ops == nullptr ||
        state->device->ops->download == nullptr) {
      result = Status::fail(Reason::TransferInvalid);
    } else {
      const DownloadResult transfer = state->device->ops->download(
          *state->device, *observed_buffer, data, bytes);
      result = transfer.status;
      state->read_staging_bytes = ::rund::detail::counter::SaturatingAdd(
          state->read_staging_bytes, transfer.staging_bytes);
      state->read_staging_reused = ::rund::detail::counter::SaturatingAdd(
          state->read_staging_reused, transfer.staging_reused_bytes);
      state->read_staging_peak =
          std::max(state->read_staging_peak, transfer.staging_peak_bytes);
      state->read_staging_budget =
          std::max(state->read_staging_budget, transfer.staging_budget);
      ::rund::detail::counter::Accumulate(state->stats.buffer_allocations,
                                          transfer.buffer_allocations);
      ::rund::detail::counter::Accumulate(state->stats.buffer_reuses,
                                          transfer.buffer_reuses);
      ::rund::detail::counter::Accumulate(state->stats.command_submits,
                                          transfer.command_submits);
      ::rund::detail::counter::Accumulate(state->stats.readback_ns,
                                          transfer.readback_ns);
      if (result && !transfer.payload_hash_valid) {
        result = Status::fail(Reason::TransferInvalid);
      }
      if (result) {
        leaf_hash = transfer.payload_hash;
      }
    }
  }
  if (result) {
    output.hash = leaf_hash;
    bool completed = false;
    // A successful run opens one canonical observation epoch by setting the
    // remaining-output count. Transactional parity changes only the physical
    // Buffer selected above; it must not create or suppress an output.
    // Pre-run and freshly restored state remains readable without being
    // attributed to a Pipeline execution because it has no open epoch.
    if (!output.observed && state->unobserved_outputs != 0u) {
      output.observed = true;
      completed = --state->unobserved_outputs == 0u;
    }
    if (bytes != 0u) {
      ::rund::detail::counter::Accumulate(state->stats.download_events, 1u);
      ::rund::detail::counter::Accumulate(state->stats.downloaded_bytes, bytes);
      state->read_transfer_bytes = ::rund::detail::counter::SaturatingAdd(
          state->read_transfer_bytes, bytes);
      state->read_transfer_peak =
          std::max<std::uint64_t>(state->read_transfer_peak, bytes);
      record_transfer(*state->device, bytes);
    }
    if (completed) {
      std::uint64_t hash = ::rund::node::hash_detail::kFnvOffset;
      mix(hash, state->outputs.size());
      for (const PipelineOutputState &observed : state->outputs) {
        if (!observed.observed ||
            observed.resource >= state->resources.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const PipelineResource &value = state->resources[observed.resource];
        mix(hash, observed.resource);
        mix(hash, static_cast<std::uint64_t>(value.type));
        mix(hash, value.format.integer_bits);
        mix(hash, value.format.fraction_bits);
        mix(hash, static_cast<std::uint64_t>(value.format.rounding));
        mix(hash, static_cast<std::uint64_t>(value.format.overflow));
        mix(hash, static_cast<std::uint64_t>(value.format.approximation));
        mix(hash, value.count);
        mix(hash, observed.hash);
      }
      state->stats.output_hash = hash;
    }
  }
  return result;
}

} // namespace rund::compute::detail
