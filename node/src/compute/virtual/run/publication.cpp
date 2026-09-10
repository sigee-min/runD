#include "publication.hpp"

#include "transaction.hpp"

#include "../../pipeline/local.hpp"
#include "../state.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <mutex>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool valid_cursor_bank(
    const PipelineState &pipeline, const VirtualRunPublicationCursor &cursor,
    const std::size_t bank) noexcept {
  if (bank >= VirtualRunPublicationCursor::BankCount ||
      !has_private_residency_authority(pipeline) || pipeline.transactional ||
      pipeline.residency_pool == nullptr || pipeline.residency == nullptr ||
      pipeline.residency_stage != PipelineResidencyStage::Direct ||
      pipeline.residency_graph_stage != residency::NoGraphStage ||
      pipeline.residency_port_count != 0u || pipeline.residency_bank != bank ||
      pipeline.residency_input == std::numeric_limits<std::uint32_t>::max() ||
      pipeline.residency_output == std::numeric_limits<std::uint32_t>::max() ||
      pipeline.residency_input == pipeline.residency_output ||
      pipeline.phase != PipelinePhase::Ready || pipeline.publication == nullptr ||
      pipeline.publication->attempt_active ||
      pipeline.publication->device_lost || pipeline.control_poisoned) {
    return false;
  }
  const std::uint64_t base_generation = cursor.base_generation[bank];
  const std::uint64_t base_payload_epoch = cursor.base_payload_epoch[bank];
  const std::uint64_t terminal_count = cursor.terminal_count[bank];
  const std::uint64_t control_generation = cursor.control_generation[bank];
  if (base_generation >= PipelineGenerationCapacity ||
      base_payload_epoch == std::numeric_limits<std::uint64_t>::max() ||
      cursor.base_parity[bank] > 1u ||
      terminal_count == std::numeric_limits<std::uint64_t>::max() ||
      control_generation == std::numeric_limits<std::uint64_t>::max() ||
      terminal_count > std::numeric_limits<std::uint64_t>::max() -
                            base_generation ||
      control_generation != base_generation + terminal_count ||
      pipeline.publication->generation != base_generation ||
      pipeline.publication->payload_epoch != base_payload_epoch ||
      pipeline.publication->parity != cursor.base_parity[bank] ||
      pipeline.stats.publication.generation != base_generation ||
      pipeline.native_generation != control_generation ||
      pipeline.native_parity != cursor.base_parity[bank]) {
    return false;
  }
  return true;
}

[[nodiscard]] Status poison_cursor_banks(
    std::array<PipelineState *, VirtualRunPublicationCursor::BankCount> banks,
    VirtualRunPublicationCursor &cursor, const bool locked) noexcept {
  std::sort(banks.begin(), banks.end(), std::less<PipelineState *>{});
  const bool shape = banks[0u] != nullptr && banks[1u] != nullptr &&
                     banks[0u] != banks[1u];
  if (!locked) {
    std::array<std::unique_lock<std::mutex>,
               VirtualRunPublicationCursor::BankCount>
        locks{};
    std::size_t lock_count = 0u;
    for (PipelineState *const pipeline : banks) {
      if (pipeline == nullptr ||
          (lock_count != 0u && banks[lock_count - 1u] == pipeline)) {
        continue;
      }
      locks[lock_count++] = std::unique_lock<std::mutex>{pipeline->gate};
    }
    const bool valid =
        shape && banks[0u]->publication != nullptr &&
        banks[1u]->publication != nullptr &&
        banks[0u]->publication != banks[1u]->publication &&
        banks[0u]->device != nullptr &&
        banks[0u]->device == banks[1u]->device;
    for (PipelineState *const pipeline : banks) {
      if (pipeline == nullptr) {
        continue;
      }
      pipeline->control_poisoned = true;
      pipeline->failure = Reason::PipelinePoisoned;
      pipeline->phase = PipelinePhase::Poisoned;
    }
    cursor.active = false;
    return valid ? Status::success() : Status::fail(Reason::PipelinePoisoned);
  }
  for (PipelineState *const pipeline : banks) {
    if (pipeline == nullptr) {
      continue;
    }
    pipeline->control_poisoned = true;
    pipeline->failure = Reason::PipelinePoisoned;
    pipeline->phase = PipelinePhase::Poisoned;
  }
  cursor.active = false;
  return shape ? Status::success() : Status::fail(Reason::PipelinePoisoned);
}

} // namespace

Status begin_virtual_run_publication_cursor(
    VirtualPipelineState &state,
    VirtualRunTransaction &transaction) noexcept {
  transaction.cursor = VirtualRunPublicationCursor{};
  if (state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline == state.alternate_pipeline ||
      state.pipeline->publication == nullptr ||
      state.alternate_pipeline->publication == nullptr ||
      state.pipeline->publication == state.alternate_pipeline->publication ||
      state.pipeline->device == nullptr ||
      state.pipeline->device != state.alternate_pipeline->device) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<PipelineState *, VirtualRunPublicationCursor::BankCount> banks{
      state.pipeline.get(), state.alternate_pipeline.get()};
  std::array<PipelinePublicationState *, VirtualRunPublicationCursor::BankCount>
      publications{state.pipeline->publication.get(),
                   state.alternate_pipeline->publication.get()};
  std::sort(banks.begin(), banks.end(), std::less<PipelineState *>{});
  std::sort(publications.begin(), publications.end(),
            std::less<PipelinePublicationState *>{});
  std::array<std::unique_lock<std::mutex>,
             VirtualRunPublicationCursor::BankCount>
      pipeline_locks{};
  for (std::size_t index = 0u; index < banks.size(); ++index) {
    pipeline_locks[index] = std::unique_lock<std::mutex>{banks[index]->gate};
  }
  std::array<std::unique_lock<std::mutex>,
             VirtualRunPublicationCursor::BankCount>
      publication_locks{};
  for (std::size_t index = 0u; index < publications.size(); ++index) {
    publication_locks[index] =
        std::unique_lock<std::mutex>{publications[index]->gate};
  }

  VirtualRunPublicationCursor cursor{};
  for (std::size_t bank = 0u;
       bank < VirtualRunPublicationCursor::BankCount; ++bank) {
    PipelineState &pipeline =
        bank == 0u ? *state.pipeline : *state.alternate_pipeline;
    if (!has_private_residency_authority(pipeline) || pipeline.transactional ||
        pipeline.residency_pool == nullptr || pipeline.residency == nullptr ||
        pipeline.residency_stage != PipelineResidencyStage::Direct ||
        pipeline.residency_graph_stage != residency::NoGraphStage ||
        pipeline.residency_port_count != 0u ||
        pipeline.residency_bank != bank || pipeline.publications.size() != 0u ||
        pipeline.windows.size() != 0u ||
        pipeline.residency_input == std::numeric_limits<std::uint32_t>::max() ||
        pipeline.residency_output == std::numeric_limits<std::uint32_t>::max() ||
        pipeline.residency_input == pipeline.residency_output) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (pipeline.phase != PipelinePhase::Ready ||
        pipeline.publication->attempt_active) {
      return Status::fail(Reason::PipelineBusy);
    }
    if (pipeline.publication->device_lost) {
      return Status::fail(Reason::DeviceLost);
    }
    const std::uint64_t generation = pipeline.publication->generation;
    const std::uint64_t payload_epoch = pipeline.publication->payload_epoch;
    const std::uint8_t parity = pipeline.publication->parity;
    if (generation >= PipelineGenerationCapacity ||
        payload_epoch == std::numeric_limits<std::uint64_t>::max() ||
        parity > 1u || pipeline.stats.publication.generation != generation) {
      return Status::fail(Reason::PipelineCapacity);
    }
    cursor.base_generation[bank] = generation;
    cursor.base_payload_epoch[bank] = payload_epoch;
    cursor.base_parity[bank] = parity;
    cursor.control_generation[bank] = generation;
  }
  cursor.active = true;
  transaction.cursor = cursor;
  return Status::success();
}

Status record_virtual_run_publication_terminal(
    VirtualRunTransaction &transaction, const std::size_t bank) noexcept {
  if (!transaction.cursor.active ||
      bank >= VirtualRunPublicationCursor::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint64_t &control = transaction.cursor.control_generation[bank];
  std::uint64_t &count = transaction.cursor.terminal_count[bank];
  if (control == std::numeric_limits<std::uint64_t>::max() ||
      count == std::numeric_limits<std::uint64_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  ++control;
  ++count;
  return Status::success();
}

Status poison_virtual_run_publication_cursor(
    VirtualPipelineState &state, VirtualRunPublicationCursor &cursor) noexcept {
  return poison_cursor_banks(
      std::array<PipelineState *, VirtualRunPublicationCursor::BankCount>{
          state.pipeline.get(), state.alternate_pipeline.get()},
      cursor, false);
}

Status abort_virtual_run_publication_cursor(
    VirtualPipelineState &state, VirtualRunPublicationCursor &cursor) noexcept {
  if (!cursor.active) {
    return Status::success();
  }
  std::array<PipelineState *, VirtualRunPublicationCursor::BankCount> banks{
      state.pipeline.get(), state.alternate_pipeline.get()};
  const auto poison = [&]() noexcept {
    return poison_cursor_banks(banks, cursor, false);
  };
  if (state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline == state.alternate_pipeline ||
      state.pipeline->publication == nullptr ||
      state.alternate_pipeline->publication == nullptr ||
      state.pipeline->publication == state.alternate_pipeline->publication ||
      state.pipeline->device == nullptr ||
      state.pipeline->device != state.alternate_pipeline->device) {
    return poison();
  }

  std::sort(banks.begin(), banks.end(), std::less<PipelineState *>{});
  std::array<std::unique_lock<std::mutex>,
             VirtualRunPublicationCursor::BankCount>
      pipeline_locks{};
  for (std::size_t index = 0u; index < banks.size(); ++index) {
    pipeline_locks[index] = std::unique_lock<std::mutex>{banks[index]->gate};
  }
  std::array<PipelinePublicationState *, VirtualRunPublicationCursor::BankCount>
      publications{state.pipeline->publication.get(),
                   state.alternate_pipeline->publication.get()};
  std::sort(publications.begin(), publications.end(),
            std::less<PipelinePublicationState *>{});
  std::array<std::unique_lock<std::mutex>,
             VirtualRunPublicationCursor::BankCount>
      publication_locks{};
  for (std::size_t index = 0u; index < publications.size(); ++index) {
    publication_locks[index] =
        std::unique_lock<std::mutex>{publications[index]->gate};
  }

  for (std::size_t bank = 0u;
       bank < VirtualRunPublicationCursor::BankCount; ++bank) {
    PipelineState &pipeline =
        bank == 0u ? *state.pipeline : *state.alternate_pipeline;
    if (!valid_cursor_bank(pipeline, cursor, bank)) {
      (void)poison_cursor_banks(banks, cursor, true);
      return Status::fail(Reason::PipelinePoisoned);
    }
  }
  for (std::size_t bank = 0u;
       bank < VirtualRunPublicationCursor::BankCount; ++bank) {
    PipelineState &pipeline =
        bank == 0u ? *state.pipeline : *state.alternate_pipeline;
    const Status seeded = seed_pipeline_generations(
        pipeline, cursor.base_generation[bank], cursor.base_parity[bank]);
    if (!seeded || pipeline.native_generation != cursor.base_generation[bank] ||
        pipeline.native_parity != cursor.base_parity[bank]) {
      (void)poison_cursor_banks(banks, cursor, true);
      return Status::fail(Reason::PipelinePoisoned);
    }
    cursor.control_generation[bank] = cursor.base_generation[bank];
    cursor.terminal_count[bank] = 0u;
  }
  cursor.active = false;
  return Status::success();
}

} // namespace rund::compute::detail
