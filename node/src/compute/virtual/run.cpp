#include "backing.hpp"
#include "local.hpp"
#include "run/backing.hpp"
#include "run/evidence.hpp"
#include "run/projection.hpp"
#include "run/wave.hpp"

#include "../../hash/fnv.hpp"

#include <mutex>

namespace rund::compute::detail {

Status run_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  return run_virtual_pipeline(state, state == nullptr || state->input == nullptr
                                         ? 0u
                                         : state->input->count);
}

Status run_virtual_pipeline(const std::shared_ptr<VirtualPipelineState> &state,
                            const std::uint64_t active_count) noexcept {
  if (!valid_virtual_pipeline(state)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::unique_lock state_lock{state->gate, std::try_to_lock};
  if (!state_lock.owns_lock() ||
      state->phase == VirtualPipelinePhase::Running) {
    return Status::fail(Reason::PipelineBusy);
  }
  if (state->phase == VirtualPipelinePhase::Poisoned) {
    return Status::fail(Reason::PipelinePoisoned);
  }
  if (active_count > state->input->count ||
      active_count > state->output->count) {
    return Status::fail(Reason::ShapeMismatch);
  }
  state->phase = VirtualPipelinePhase::Running;

  Stats stats = begin_virtual_run_evidence(*state, active_count);
  std::uint64_t failed_page = ResidencyStats::no_failed_page;
  bool poison_pipeline = false;
  const auto finish = [&](const Status status,
                          const std::uint64_t output_hash = 0u) noexcept {
    return publish_virtual_run_evidence(*state, stats, status, failed_page,
                                        output_hash, poison_pipeline);
  };

  VirtualBacking &input_backing = *state->input->backing;
  VirtualBacking &output_backing = *state->output->backing;
  std::scoped_lock backing_locks{VirtualBackingAccess::gate(input_backing),
                                 VirtualBackingAccess::gate(output_backing)};

  VirtualRunProjection run{};
  if (!project_virtual_run(*state, active_count, run)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  stats.pipeline.residency.active_slots_peak = run.active.active_slots_peak;
  const Status recovery = validate_virtual_recovery(
      input_backing, output_backing, run.active.output_bytes);
  if (!recovery) {
    return finish(recovery);
  }

  ::rund::node::hash_detail::Fnv output_hash{};
  if (active_count == 0u) {
    return finish(Status::success(), output_hash.Finish());
  }
  if (!bind_virtual_run_transfer(*state, run)) {
    poison_pipeline = true;
    return finish(Status::fail(Reason::PipelineInvalid));
  }
  for (std::uint64_t wave = 0u; wave < run.active.linear.wave_count(); ++wave) {
    const VirtualWaveResult result = execute_virtual_wave(
        *state, input_backing, output_backing, run, wave, stats, output_hash);
    if (!result.status) {
      failed_page = result.failed_page;
      poison_pipeline = result.poison_pipeline;
      return finish(result.status);
    }
  }

  clear_virtual_recovery(output_backing);
  return finish(Status::success(), output_hash.Finish());
}

} // namespace rund::compute::detail
