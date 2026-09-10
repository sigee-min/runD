#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "projection.hpp"

#include <array>

namespace rund::compute::detail::graph_reduce {

VirtualGraphResult execute_tiled_graph(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking *const output, const VirtualRunProjection &run, Stats &stats,
    VirtualReduction *const reduction) noexcept {
  residency::Pool &pool = *state.pipeline->residency_pool;
  residency::Authority &authority = pool.authority();
  const bool cpu = state.pipeline->device->backend == Backend::Cpu;
  if (cpu && !cpu_graph_ready(state)) {
    return failed(Status::fail(Reason::PipelineBusy), 0u);
  }
  state.failure_log.reset();
  const residency::TiledGraphPlan &graph =
      state.pipeline->residency->tiled_graph();
  const bool persisted_output = output != nullptr;
  if (persisted_output == (reduction != nullptr) ||
      graph.stages().size() < 2u || graph.stages().front().ports.size() < 2u ||
      graph.stages().back().ports.size() < 2u) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    state.failure_log.note(Fail::NoStage, 0u, Phase::Prepare, Check::Prepare,
                           failure);
    return failed(failure, 0u);
  }
  const std::size_t terminal_stage = graph.stages().size() - 1u;
  const std::uint64_t identity = graph_identity(run);
  const std::uint64_t batches = run.active.graph.batch_count();
  const std::uint32_t capacity = static_cast<std::uint32_t>(run.frame_capacity);
  std::array<Ticket, residency::Pool::BankCount> tickets{};
  Wavefront wavefront{};
  if (!wavefront.reset(run.active.graph, graph)) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    state.failure_log.note(Fail::NoStage, 0u, Phase::Prepare, Check::Prepare,
                           failure);
    return failed(failure, 0u);
  }
  if (inputs.empty() || inputs.front() == nullptr) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    state.failure_log.note(Fail::NoStage, 0u, Phase::Prepare, Check::Prepare,
                           failure);
    return failed(failure, 0u);
  }
  PrefetchController prefetch{state, inputs, run,       stats,
                              pool,  graph,  wavefront, batches};
  const bool pair_route = prefetch.pair_supported();
  SupplyController supply{
      run, stats, pool, graph, wavefront, prefetch, state.failure_log};
  supply.configure_retry(state.pipeline->residency->identity(),
                         cpu && persisted_output);
  StageController stages{stats, pool, wavefront, identity, terminal_stage};
  CollectiveController collective{
      run,       stats,  pool,           graph,
      wavefront, stages, terminal_stage, state.failure_log};
  MiddleController middle{state,    run,       stats,         pool,
                          graph,    wavefront, prefetch,      stages,
                          identity, capacity,  terminal_stage};
  ::rund::node::hash_detail::Fnv output_hash{};
  PersistController persists{
      authority,      pool,       state.pipeline->residency,
      output,         run,        graph,
      stats,          wavefront,  stages,
      terminal_stage, output_hash};
  AbortController abort{authority,
                        tickets,
                        prefetch,
                        persists,
                        stages,
                        wavefront,
                        state.cpu_receipts,
                        state.cpu_quarantine_hold,
                        state.failure_log};
  std::span<Ticket> ticket_span{tickets};
  GraphExecutionContext context{
      state,      inputs,   output,         run,         stats,
      reduction,  pool,     authority,      graph,       ticket_span,
      wavefront,  prefetch, supply,         stages,      collective,
      middle,     persists, abort,          output_hash, identity,
      batches,    capacity, terminal_stage, cpu,         persisted_output,
      pair_route, false,    false,
  };

  const VirtualGraphResult initial = prepare_initial(context);
  if (!initial.status) {
    return initial;
  }
  const VirtualGraphResult batches_result = execute_batches(context);
  if (!batches_result.status) {
    return batches_result;
  }
  return finish_tiled(context);
}

} // namespace rund::compute::detail::graph_reduce
