#include "contract.hpp"
#include "program.hpp"
#include "report.hpp"

#include "../../process.hpp"
#include "allocation.hpp"

#include <chrono>
#include <utility>

namespace rund::measure::compute {
using namespace preparation_memory;

bool MeasurePreparationMemory(const Backend backend, const bool materialize) {
  using namespace ::rund::compute;
  PreparationMemoryObservation observed{};
  auto device = open(TargetFor(backend));
  if (!device) {
    CaptureFailure(observed, device);
    observed.status =
        observed.code == Code::Unavailable ? "unavailable" : "open_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  auto seed = LargeSeedProgram<Maximum, Tile>(*device);
  auto action = ActionProgram(*device);
  auto fold = FoldProgram(*device);
  auto second_seed = SecondSeedProgram<Maximum, Tile>(*device);
  auto second_action = SecondActionProgram(*device);
  auto second_fold = SecondFoldProgram(*device);
  auto consume = ConsumeProgram(*device);
  auto recurrence = OrdinaryRecurrenceProgram(*device);
  auto publish = PublishProgram(*device);
  if (!seed || !action || !fold || !second_seed || !second_action ||
      !second_fold || !consume || !recurrence || !publish) {
    if (!seed) {
      CaptureFailure(observed, seed);
      observed.status = "seed_compile_failed";
    } else if (!action) {
      CaptureFailure(observed, action);
      observed.status = "action_compile_failed";
    } else if (!fold) {
      CaptureFailure(observed, fold);
      observed.status = "fold_compile_failed";
    } else if (!second_seed) {
      CaptureFailure(observed, second_seed);
      observed.status = "second_seed_compile_failed";
    } else if (!second_action) {
      CaptureFailure(observed, second_action);
      observed.status = "second_action_compile_failed";
    } else if (!second_fold) {
      CaptureFailure(observed, second_fold);
      observed.status = "second_fold_compile_failed";
    } else if (!consume) {
      CaptureFailure(observed, consume);
      observed.status = "consume_compile_failed";
    } else if (!recurrence) {
      CaptureFailure(observed, recurrence);
      observed.status = "recurrence_compile_failed";
    } else {
      CaptureFailure(observed, publish);
      observed.status = "publish_compile_failed";
    }
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  auto outer = device->buffer<std::uint32_t>(1u);
  auto queue = device->buffer<std::uint32_t>(Maximum);
  auto domain = device->buffer<std::uint32_t>(Domain);
  auto count = device->buffer<std::uint32_t>(1u);
  auto first_output = device->buffer<std::uint32_t>(1u);
  auto first_window = device->buffer<std::uint32_t>(Maximum);
  auto consumed = device->buffer<std::uint32_t>(Maximum);
  auto recurrence_output = device->buffer<std::uint32_t>(Maximum);
  auto second_output = device->buffer<std::uint32_t>(1u);
  auto published_result = device->buffer<std::uint32_t>(1u);
  auto pending_result = device->buffer<std::uint32_t>(1u);
  if (!outer || !queue || !domain || !count || !first_output || !first_window ||
      !consumed || !recurrence_output || !second_output || !published_result ||
      !pending_result) {
    if (!outer) {
      CaptureFailure(observed, outer);
    } else if (!queue) {
      CaptureFailure(observed, queue);
    } else if (!domain) {
      CaptureFailure(observed, domain);
    } else if (!count) {
      CaptureFailure(observed, count);
    } else if (!first_output) {
      CaptureFailure(observed, first_output);
    } else if (!first_window) {
      CaptureFailure(observed, first_window);
    } else if (!consumed) {
      CaptureFailure(observed, consumed);
    } else if (!recurrence_output) {
      CaptureFailure(observed, recurrence_output);
    } else if (!second_output) {
      CaptureFailure(observed, second_output);
    } else if (!published_result) {
      CaptureFailure(observed, published_result);
    } else {
      CaptureFailure(observed, pending_result);
    }
    observed.status = observed.code == Code::Unavailable
                          ? "unavailable"
                          : "buffer_setup_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  const auto first_body = tile_repeat<Inner>(*seed, *action, *fold);
  const auto second_body =
      tile_repeat<SecondInner>(*second_seed, *second_action, *second_fold);
  const auto make_builder = [&]() {
    auto candidate = pipeline(*device);
    candidate.state(*published_result, *pending_result)
        .windows<Maximum, Tile>(
            first_body, window(*count), read(*outer, *queue, *domain),
            write_final(*first_output), write_window(*first_window))
        .then(*consume, read(*first_window), write(*consumed))
        .repeat<OrdinaryIterations>(*recurrence, read(*consumed),
                                    write_final(*recurrence_output))
        .windows<Maximum, Tile>(
            second_body, window(*count),
            read(*first_output, *recurrence_output, *domain),
            write_final(*second_output))
        .then(*publish, read(*second_output, *published_result),
              write(*pending_result))
        .commit();
    return candidate;
  };
  auto builder = make_builder();
  observed.plan_current_rss_before = ProcessCurrentResidentBytes();
  observed.plan_rss_before = ProcessMaximumResidentBytes();
  const auto plan_begin = Clock::now();
  const auto planned = builder.plan();
  const auto plan_end = Clock::now();
  observed.plan_current_rss_after = ProcessCurrentResidentBytes();
  observed.plan_rss_after = ProcessMaximumResidentBytes();
  observed.plan_wall_us =
      std::chrono::duration<double, std::micro>(plan_end - plan_begin).count();
  if (!planned) {
    CaptureFailure(observed, planned);
    observed.status = "plan_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }
  observed.plan = *planned;
  observed.plan_contract = PlanContract(observed.plan, backend);
  if (!observed.plan_contract) {
    observed.status = "plan_contract_failed";
    PrintObservation(backend, materialize, observed);
    return false;
  }

  const auto check_short_budget = [&]() {
    observed.short_budget_checked = true;
    auto short_builder = make_builder();
    const auto short_plan = short_builder.plan();
    if (!short_plan || *short_plan != observed.plan ||
        observed.plan.peak_bytes == 0u) {
      return false;
    }
    const MemoryStats before = device->memory();
    auto rejected =
        std::move(short_builder)
            .budget(MemoryBudget{.bytes = observed.plan.peak_bytes - 1u})
            .prepare();
    const MemoryStats after = device->memory();
    observed.short_budget_reason = rejected.reason();
    observed.short_budget_rejected =
        !rejected && rejected.reason() == Reason::PipelineMemoryBudget;
    observed.short_budget_no_allocation = NoAllocation(before, after);
    return observed.short_budget_rejected &&
           observed.short_budget_no_allocation;
  };

  if (!materialize) {
    observed.contract =
        PlanObservationContract(observed) && check_short_budget();
    observed.status = observed.contract ? "plan_ok" : "plan_contract_failed";
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  observed.prepare_attempted = true;
  observed.prepare_current_rss_before = ProcessCurrentResidentBytes();
  observed.prepare_rss_before = ProcessMaximumResidentBytes();
  node_compute_allocation::Start();
  const auto begin = Clock::now();
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = observed.plan.peak_bytes})
                      .prepare();
  const auto end = Clock::now();
  node_compute_allocation::Stop();
  observed.prepare_host_allocation_count = node_compute_allocation::Count();
  observed.prepare_host_allocation_bytes = node_compute_allocation::Bytes();
  observed.prepare_current_rss_after = ProcessCurrentResidentBytes();
  observed.prepare_rss_after = ProcessMaximumResidentBytes();
  observed.prepare_wall_us =
      std::chrono::duration<double, std::micro>(end - begin).count();
  observed.prepare_rss_within_committed_peak = WithinAdditionalResidentBytes(
      observed.prepare_current_rss_before, observed.prepare_current_rss_after,
      observed.prepare_rss_before, observed.prepare_rss_after,
      observed.plan.committed_peak_bytes);
  if (!prepared) {
    CaptureFailure(observed, prepared);
    const bool short_budget_contract = check_short_budget();
    if (observed.code == Code::Unavailable) {
      observed.status = "unavailable";
      observed.contract = PlanObservationContract(observed) &&
                          short_budget_contract &&
                          observed.prepare_rss_within_committed_peak;
      PrintObservation(backend, materialize, observed);
      return observed.contract;
    }
    observed.precise_failure = PreciseFailure(observed);
    observed.status =
        observed.precise_failure ? "precise_failure" : "imprecise_failure";
    observed.contract = PlanObservationContract(observed) &&
                        short_budget_contract && observed.precise_failure &&
                        observed.prepare_rss_within_committed_peak;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  observed.prepare_ok = true;
  observed.plan_frozen = prepared->plan() == observed.plan;
  observed.memory = prepared->memory();
  observed.memory_contract =
      CaptureLargestRetainedGroup(observed, *prepared) &&
      PreparedMemoryContract(observed.plan, observed.memory, backend) &&
      CaptureBackendPreparation(observed, *prepared, backend);
  const bool short_budget_contract = check_short_budget();
  const bool prepared_contract =
      PlanObservationContract(observed) && short_budget_contract &&
      observed.plan_frozen && observed.memory_contract &&
      observed.prepare_wall_us > 0.0 && observed.prepare_rss_before != 0u &&
      observed.prepare_rss_after >= observed.prepare_rss_before;
  observed.contract =
      prepared_contract && observed.prepare_rss_within_committed_peak;
  observed.status =
      observed.contract
          ? "prepare_ok"
          : (prepared_contract && !observed.prepare_rss_within_committed_peak
                 ? "process_rss_contract_failed"
                 : "prepare_contract_failed");
  PrintObservation(backend, materialize, observed);
  return observed.contract;
}


} // namespace rund::measure::compute
