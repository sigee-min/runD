#include "../internal.hpp"

#include "rebase.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

struct BankSnapshot final {
  std::uint64_t generation{};
  std::uint8_t parity{};
  std::uint64_t native_generation{};
  std::uint8_t native_parity{};
  bool control_poisoned{};
};

struct RebaseSnapshot final {
  std::array<BankSnapshot, residency::execution::BankCapacity> banks{};
};

[[nodiscard]] RebaseResult poisoned() noexcept {
  return RebaseResult{Status::fail(Reason::PipelinePoisoned),
                      RebaseDisposition::Poisoned};
}

[[nodiscard]] bool capture_snapshot(SlidingProductRun &state,
                                    RebaseSnapshot &snapshot) noexcept {
  if (state.cold == nullptr ||
      state.cold->pipelines.size() != residency::execution::BankCapacity) {
    return false;
  }
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    const std::shared_ptr<PipelineState> &owner = state.cold->pipelines[bank];
    if (owner == nullptr || !state.pipeline_started[bank]) {
      return false;
    }
    PipelineState &pipeline = *owner;
    std::lock_guard lock{pipeline.gate};
    if (pipeline.phase != PipelinePhase::Running ||
        pipeline.publication == nullptr) {
      return false;
    }
    const BankSnapshot saved{
        .generation = state.cold->snapshot.generation[bank],
        .parity = state.cold->snapshot.parity[bank],
        .native_generation = pipeline.native_generation,
        .native_parity = pipeline.native_parity,
        .control_poisoned = pipeline.control_poisoned,
    };
    if (saved.native_generation != saved.generation ||
        saved.native_parity != saved.parity) {
      return false;
    }
    snapshot.banks[bank] = saved;
    if (saved.control_poisoned) {
      return false;
    }
    std::lock_guard publication_lock{pipeline.publication->gate};
    if (!pipeline.publication->attempt_active ||
        pipeline.publication->generation != saved.generation ||
        pipeline.publication->parity != saved.parity ||
        pipeline.attempt.generation != saved.generation ||
        pipeline.attempt.parity != saved.parity) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool restore_snapshot(SlidingProductRun &state,
                                    const RebaseSnapshot &snapshot) noexcept {
  bool restored = true;
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    PipelineState &pipeline = *state.cold->pipelines[bank];
    std::lock_guard lock{pipeline.gate};
    const BankSnapshot &saved = snapshot.banks[bank];
    const Status seeded =
        seed_pipeline_generations(pipeline, saved.generation, saved.parity);
    const bool exact = seeded &&
                       pipeline.native_generation == saved.native_generation &&
                       pipeline.native_parity == saved.native_parity;
    if (!exact) {
      pipeline.control_poisoned = true;
      restored = false;
      continue;
    }
    // A successful rollback may clear only poison introduced by this attempt.
    pipeline.control_poisoned = saved.control_poisoned;
  }
  if (!restored) {
    for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
      std::lock_guard lock{state.cold->pipelines[bank]->gate};
      state.cold->pipelines[bank]->control_poisoned = true;
    }
  }
  return restored;
}

[[nodiscard]] bool advance_snapshot(SlidingProductRun &state,
                                    const RebaseSnapshot &snapshot) noexcept {
  for (std::size_t bank = 0u; bank < state.cold->pipelines.size(); ++bank) {
    const BankSnapshot &saved = snapshot.banks[bank];
    if (saved.generation >= PipelineGenerationCapacity) {
      return false;
    }
    PipelineState &pipeline = *state.cold->pipelines[bank];
    std::lock_guard lock{pipeline.gate};
    std::uint8_t parity = saved.parity;
    if (pipeline.transactional) {
      parity ^= 1u;
    }
    const Status seeded =
        seed_pipeline_generations(pipeline, saved.generation + 1u, parity);
    if (!seeded) {
      pipeline.control_poisoned = true;
      return false;
    }
  }
  return true;
}

} // namespace

RebaseResult rebase_pipeline_sliding(SlidingProductRun &state,
                                     const Status result) noexcept {
  RebaseSnapshot snapshot{};
  if (!capture_snapshot(state, snapshot)) {
    return poisoned();
  }

  const bool advance = static_cast<bool>(result);
  if (advance && !advance_snapshot(state, snapshot)) {
    if (restore_snapshot(state, snapshot)) {
      return RebaseResult{Status::fail(Reason::PipelineInvalid),
                          RebaseDisposition::RestoredFailure};
    }
    return poisoned();
  }
  if (!advance && !restore_snapshot(state, snapshot)) {
    return poisoned();
  }
  return RebaseResult{result, advance ? RebaseDisposition::Ready
                                      : RebaseDisposition::RestoredFailure};
}

} // namespace rund::compute::detail::sliding_product_detail
