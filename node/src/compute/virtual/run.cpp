#include "backing.hpp"
#include "local.hpp"
#include "run/admission.hpp"
#include "run/backing.hpp"
#include "run/backing_set.hpp"
#include "run/cache.hpp"
#include "run/device_vsm/model.hpp"
#include "run/device_vsm/operations.hpp"
#include "run/device_vsm/route.hpp"
#include "run/dispatch.hpp"
#include "run/evidence.hpp"
#include "run/final.hpp"
#include "run/projection.hpp"
#include "run/transaction.hpp"

#include <array>
#include <mutex>
#include <span>
#include <utility>

namespace rund::compute::detail {

Status run_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  const VirtualBufferState *const input =
      state == nullptr ? nullptr : virtual_input(*state);
  return run_virtual_pipeline(state, input == nullptr ? 0u : input->count);
}

Status run_virtual_pipeline(const std::shared_ptr<VirtualPipelineState> &state,
                            const std::uint64_t active_count) noexcept {
  std::unique_lock<std::mutex> state_lock;
  const VirtualRunAdmission admission = admit_virtual_run(
      state, active_count, state_lock, VirtualRunAdmissionMode::Synchronous);
  if (!admission.claim) {
    return admission.status;
  }
  state->phase = VirtualPipelinePhase::Running;
  const VirtualBufferState *const input = virtual_input(*state);

  Stats stats = begin_virtual_run_evidence(*state, active_count);
  std::uint64_t failed_page = ResidencyStats::no_failed_page;
  bool poison_pipeline = admission.poison_pipeline;
  VirtualRunTransaction transaction{};
  VirtualRunResources resources{};
  const auto finish = [&](const Status status,
                          const std::uint64_t output_hash = 0u,
                          const VirtualRunWriteCertainty certainty =
                              VirtualRunWriteCertainty::KnownNoWrite) noexcept {
    return finish_virtual_run(*state, stats, transaction, status, certainty,
                              failed_page, output_hash, poison_pipeline,
                              &resources);
  };

  if (!admission.status) {
    return finish(admission.status);
  }

  VirtualBacking &input_backing = *input->backing;
  VirtualBacking &output_backing = *state->output->backing;
  if (!lock_backings(*state, resources)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }

  VirtualRunProjection run{};
  if (!project_virtual_run(*state, active_count, run)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  std::array<VirtualBacking *, VirtualPipelineState::InputCapacity>
      input_backings{};
  for (std::size_t index = 0u; index < state->input_count; ++index) {
    input_backings[index] = virtual_input(*state, index)->backing.get();
  }
  const std::span<VirtualBacking *const> active_inputs{input_backings.data(),
                                                       state->input_count};
  const Status recovery =
      validate_virtual_run_recovery(run, active_inputs, output_backing);
  if (!recovery) {
    return finish(recovery);
  }

  const VirtualDeviceVsmCandidate candidate =
      probe_virtual_device_vsm_route(*state, active_inputs, output_backing,
                                     run);
  if (candidate.terminal()) {
    return finish(candidate.status);
  }

  VirtualRunWork work{};
  const VirtualRunDispatchResult dispatched = dispatch_virtual_route(
      *state, active_inputs, input_backing, output_backing, run, candidate,
      active_count, stats, work, transaction, resources,
      poison_pipeline);
  if (dispatched.empty) {
    return finish_virtual_empty(*state, stats, transaction, run,
                                output_backing, work.reduction, failed_page,
                                poison_pipeline, &resources);
  }
  return finish_virtual_dispatch(*state, stats, transaction, run,
                                 output_backing, work.reduction, dispatched,
                                 failed_page, poison_pipeline, &resources);
}

} // namespace rund::compute::detail
