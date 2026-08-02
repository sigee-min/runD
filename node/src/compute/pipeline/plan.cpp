#include <rund/compute/pipeline.hpp>
#include <rund/compute/resource/plan.hpp>
#include <rund/counter.hpp>

#include "../backend.hpp"
#include "../buffer/local.hpp"
#include "../job/local.hpp"
#include "../status.hpp"
#include "claim.hpp"
#include "local.hpp"
#include "plan/contract.hpp"
#include "plan/local.hpp"
#include "plan/output.hpp"
#include "plan/prepare.hpp"
#include "state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

struct PipelinePrepareOwner final {
  // Declaration order is lifetime order: every preparation/build owner is
  // destroyed before a failed admission is refunded.
  storage::Reservation device_memory;
  std::shared_ptr<PipelineBuildState> build;
  PipelinePrepare preparation;
};

[[nodiscard]] bool exact_publication_ticket(
    const storage::Reservation &ticket,
    const std::uint64_t expected_bytes) noexcept {
  const storage::Usage usage = ticket.usage();
  return ticket.committed() &&
         ticket.max_allocated_bytes() == expected_bytes &&
         usage.physical_bytes == 0u && usage.allocated_bytes == expected_bytes;
}

[[nodiscard]] Status attach_pipeline_memory(
    PipelineState &state, storage::Reservation &total,
    const std::uint64_t publication_bytes,
    const PipelinePublicationState *const prepared_publication) noexcept {
  if (state.publication == nullptr || !total || total.committed() ||
      publication_bytes > total.max_allocated_bytes()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const bool adopted = state.publication.get() != prepared_publication;
  if (adopted) {
    if (!exact_publication_ticket(state.publication->publication_memory,
                                  publication_bytes)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else if (state.publication->publication_memory) {
    return Status::fail(Reason::PipelineInvalid);
  }

  storage::Reservation publication = total.partition(publication_bytes);
  if (!publication) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (adopted) {
    // The adopted authority already owns this exact charge. Drop only the
    // duplicate share from the new conservative admission.
    if (!publication.refund()) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    const storage::Status committed = publication.commit(storage::Usage{
        .physical_bytes = 0u,
        .allocated_bytes = publication_bytes,
    });
    if (!committed) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state.publication->publication_memory = std::move(publication);
  }

  const std::uint64_t private_bytes = total.max_allocated_bytes();
  const storage::Status committed = total.commit(storage::Usage{
      .physical_bytes = 0u,
      .allocated_bytes = private_bytes,
  });
  if (!committed) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state.private_memory = std::move(total);
  return Status::success();
}

} // namespace

Result<PipelinePlan>
plan_pipeline(const std::shared_ptr<PipelineBuildState> &build) noexcept {
  if (build == nullptr) {
    return Result<PipelinePlan>::fail(Reason::PipelineInvalid);
  }
  if (build->failure != Reason::Ok) {
    return Result<PipelinePlan>::fail(build->failure);
  }
  if (build->device == nullptr) {
    return Result<PipelinePlan>::fail(Reason::DeviceInvalid);
  }
  if (build->sealed_repetitions == 0u ||
      build->sealed_repetitions > PipelineSealedRepetitionCapacity) {
    return Result<PipelinePlan>::fail(Reason::PipelineCapacity);
  }
  if (build->steps.empty()) {
    return Result<PipelinePlan>::fail(Reason::PipelineEmpty);
  }
  if (build->memory == nullptr) {
    auto planned = plan_memory(*build);
    if (!planned) {
      return Result<PipelinePlan>::fail(planned.reason());
    }
    build->memory = std::move(planned).value();
  }
  return Result<PipelinePlan>::success(build->memory->summary);
}

void configure_pipeline_budget(const std::shared_ptr<PipelineBuildState> &build,
                               const MemoryBudget budget) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }
  build->budget = budget.bytes;
  build->has_budget = true;
}

PipelinePlan
pipeline_plan(const std::shared_ptr<PipelineState> &state) noexcept {
  return state == nullptr ? PipelinePlan{} : state->plan;
}

Result<std::shared_ptr<PipelineState>>
prepare_pipeline(std::shared_ptr<PipelineBuildState> build) noexcept {
  if (build == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  if (build->failure != Reason::Ok) {
    return Result<std::shared_ptr<PipelineState>>::fail(build->failure);
  }
  if (build->device == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(Reason::DeviceInvalid);
  }
  if (build->sealed_repetitions == 0u ||
      build->sealed_repetitions > PipelineSealedRepetitionCapacity) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  if (build->steps.empty()) {
    return Result<std::shared_ptr<PipelineState>>::fail(Reason::PipelineEmpty);
  }
  const bool nested = !build->nested_windows.empty();
  if (build->steps.size() >
          (nested ? PipelineRouteCapacity : PipelineIterationCapacity) ||
      build->binding_count >
          (nested ? PipelineRouteBindingCapacity : PipelineBindingCapacity) ||
      build->state_pairs.size() > PipelineLeafCapacity ||
      build->publications.size() > PipelineLeafCapacity) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const bool seeded = build->seed != nullptr ||
                      build->storage_seed != nullptr ||
                      build->device_seed != nullptr;
  if (build->commit != !build->state_pairs.empty() ||
      (seeded && !build->commit)) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }

  auto planned_memory = plan_pipeline(build);
  if (!planned_memory) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        planned_memory.reason());
  }
  if (build->has_budget && planned_memory->peak_bytes > build->budget) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineMemoryBudget);
  }
  const std::uint64_t committed_bytes =
      pipeline_committed_charge(*planned_memory);
  if (build->memory == nullptr ||
      build->memory->publication_committed_bytes > committed_bytes) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  const std::uint64_t publication_bytes =
      build->memory->publication_committed_bytes;
  storage::Reservation admission =
      build->device->pipeline_memory_budget.reserve(committed_bytes);
  if (!admission) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::DevicePipelineMemoryCapacity);
  }
  PipelinePrepareOwner owner{
      .device_memory = std::move(admission),
      .build = std::move(build),
      .preparation = {},
  };
  const Status materialized = materialize_pipeline(*owner.build);
  if (!materialized) {
    return Result<std::shared_ptr<PipelineState>>::fail(materialized.reason());
  }

  try {
    PipelinePrepare &preparation = owner.preparation;
    const Status admitted = admit_pipeline(owner.build, preparation);
    if (!admitted) {
      return Result<std::shared_ptr<PipelineState>>::fail(admitted.reason());
    }
    const Status scheduled = schedule_pipeline(owner.build, preparation);
    if (!scheduled) {
      return Result<std::shared_ptr<PipelineState>>::fail(scheduled.reason());
    }
    const Status bound = bind_pipeline(owner.build, preparation);
    if (!bound) {
      return Result<std::shared_ptr<PipelineState>>::fail(bound.reason(),
                                                          preparation.failure);
    }
    std::shared_ptr<PipelineState> &state = preparation.state;
    const PipelinePublicationState *const prepared_publication =
        state->publication.get();
    if (owner.build->seed != nullptr) {
      const Status restored =
          restore_pipeline_state(state, owner.build->seed);
      if (!restored) {
        return Result<std::shared_ptr<PipelineState>>::fail(restored.reason());
      }
    } else if (owner.build->storage_seed != nullptr) {
      const Status restored =
          restore_pipeline_state(state, owner.build->storage_seed);
      if (!restored) {
        return Result<std::shared_ptr<PipelineState>>::fail(restored.reason());
      }
    } else if (owner.build->device_seed != nullptr) {
      const Status restored =
          restore_pipeline_state(state, owner.build->device_seed);
      if (!restored) {
        return Result<std::shared_ptr<PipelineState>>::fail(restored.reason());
      }
    }
    const Status attached =
        attach_pipeline_memory(*state, owner.device_memory, publication_bytes,
                               prepared_publication);
    if (!attached) {
      return Result<std::shared_ptr<PipelineState>>::fail(attached.reason());
    }
    state->preparing = false;
    return Result<std::shared_ptr<PipelineState>>::success(std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
