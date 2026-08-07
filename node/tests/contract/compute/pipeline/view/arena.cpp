#include "../local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <memory>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckViewArena(rund::compute::Device &device,
                                 const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  const std::size_t expected_owners = backend == Backend::Cpu ? 0u : 1u;
  const std::uint64_t expected_bytes =
      (backend == Backend::Cpu ? 2u : 1u) * 4u * sizeof(std::int32_t);

  // Dense storage is word-addressed. Sequential 32-bit and 64-bit bindings
  // with the same byte extent share one slot, while the slot keeps the
  // strongest natural alignment required by either use.
  auto narrow_reduce =
      on(device)
          .input<std::int32_t>(4u)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto narrow_source = Upload(device, source_values);
  auto narrow_target = device.buffer<std::int32_t>(1u);
  constexpr std::array<std::int64_t, 8u> wide_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto wide_reduce =
      on(device)
          .input<std::int64_t>(2u)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto wide_source = Upload(device, wide_values);
  auto wide_target = device.buffer<std::int64_t>(1u);
  if (!narrow_reduce || !narrow_source || !narrow_target || !wide_reduce ||
      !wide_source || !wide_target) {
    return 1;
  }
  auto narrow_input = narrow_source->view(1u, 4u, 2u);
  auto wide_input = wide_source->view(1u, 2u, 2u);
  if (!narrow_input || !wide_input) {
    return 1;
  }
  auto mixed_builder =
      pipeline(device)
          .then(*narrow_reduce, read(*narrow_input), write(*narrow_target))
          .then(*wide_reduce, read(*wide_input), write(*wide_target));
  const auto mixed_plan = mixed_builder.plan();
  if (!mixed_plan ||
      mixed_plan->prepared_buffer_bytes !=
          expected_bytes + mixed_plan->scratch_bytes ||
      mixed_plan->prepared_bytes != mixed_plan->prepared_buffer_bytes +
                                        mixed_plan->prepared_host_bytes +
                                        mixed_plan->prepared_tile_bytes +
                                        mixed_plan->prepared_native_bytes) {
    return 2;
  }
  auto mixed = std::move(mixed_builder).prepare();
  const std::shared_ptr<detail::PipelineState> mixed_state =
      mixed ? detail::PipelineStateAccess::state(*mixed)
            : std::shared_ptr<detail::PipelineState>{};
  if (mixed_state == nullptr || mixed_state->steps.size() != 2u ||
      mixed_state->prepared_buffers.size() !=
          expected_owners + mixed_plan->scratch_count ||
      (expected_owners != 0u &&
       (mixed_state->steps[0u].job == nullptr ||
        mixed_state->steps[1u].job == nullptr ||
        mixed_state->steps[0u].job->workspace == nullptr ||
        mixed_state->steps[1u].job->workspace == nullptr ||
        mixed_state->steps[0u].job->workspace->arena == nullptr ||
        mixed_state->steps[0u].job->workspace->arena !=
            mixed_state->steps[1u].job->workspace->arena ||
        mixed_state->steps[0u].job->workspace->arena->binds.size() !=
            1u + mixed_plan->scratch_count ||
        mixed_state->steps[0u]
                    .job->workspace->arena->binds.refs()[0u]
                    .offset_bytes %
                alignof(std::int64_t) !=
            0u))) {
    return 3;
  }
  std::array<std::int32_t, 1u> narrow_value{};
  std::array<std::int64_t, 1u> wide_value{};
  if (!mixed || !mixed->run() ||
      !ReadExact(*mixed, *narrow_target, narrow_value) ||
      !ReadExact(*mixed, *wide_target, wide_value) ||
      narrow_value != std::array<std::int32_t, 1u>{16} ||
      wide_value != std::array<std::int64_t, 1u>{4}) {
    return 4;
  }

  // A global accelerator arena is a carrier required by every nonempty
  // Program route, even when that route has no Program chunks or local View.
  // Put the same chunk-free Map on both sides of the route that creates the
  // arena, and keep a zero-work route beside it to prove the empty gate.
  auto carrier_map =
      on(device)
          .map<std::int32_t>("pipeline-workspace-arena-carrier",
                             source_values.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto carrier_zero = on(device)
                          .map<std::int32_t>("pipeline-workspace-empty", 0u,
                                             [](auto value) { return value; })
                          .compile();
  auto carrier_source = Upload(device, source_values);
  auto carrier_before = device.buffer<std::int32_t>(source_values.size());
  auto carrier_after = device.buffer<std::int32_t>(source_values.size());
  auto carrier_reduce = device.buffer<std::int32_t>(1u);
  constexpr std::array<std::int32_t, 0u> empty_values{};
  auto carrier_empty_source = Upload(device, empty_values);
  auto carrier_empty_output = device.buffer<std::int32_t>(0u);
  if (!carrier_map || !carrier_zero || !carrier_source || !carrier_before ||
      !carrier_after || !carrier_reduce || !carrier_empty_source ||
      !carrier_empty_output) {
    return 10;
  }
  const std::shared_ptr<detail::ProgramState> &carrier_map_state =
      detail::ProgramAccess::state(*carrier_map);
  const std::shared_ptr<detail::ProgramState> &carrier_zero_state =
      detail::ProgramAccess::state(*carrier_zero);
  if (carrier_map_state == nullptr || carrier_zero_state == nullptr ||
      !carrier_map_state->chunks.empty() || !carrier_zero_state->empty()) {
    return 10;
  }
  const auto make_carrier = [&] {
    return pipeline(device)
        .then(*carrier_map, read(*carrier_source), write(*carrier_before))
        .then(*narrow_reduce, read(*narrow_input), write(*carrier_reduce))
        .then(*carrier_map, read(*carrier_source), write(*carrier_after))
        .then(*carrier_zero, read(*carrier_empty_source),
              write(*carrier_empty_output));
  };
  auto carrier_builder = make_carrier();
  const auto carrier_plan = carrier_builder.plan();
  if (!carrier_plan || carrier_plan->peak_bytes == 0u) {
    return 10;
  }
  auto below_budget =
      make_carrier()
          .budget(MemoryBudget{.bytes = carrier_plan->peak_bytes - 1u})
          .prepare();
  if (below_budget || below_budget.reason() != Reason::PipelineMemoryBudget) {
    return 11;
  }
  auto carrier = make_carrier()
                     .budget(MemoryBudget{.bytes = carrier_plan->peak_bytes})
                     .prepare();
  const std::shared_ptr<detail::PipelineState> carrier_state =
      carrier ? detail::PipelineStateAccess::state(*carrier)
              : std::shared_ptr<detail::PipelineState>{};
  if (carrier_state == nullptr || carrier->plan() != *carrier_plan ||
      carrier_state->steps.size() != 4u) {
    return 12;
  }
  for (const detail::PipelineStep &step : carrier_state->steps) {
    if (step.job == nullptr || !step.job->graph_buffers.empty()) {
      return 12;
    }
  }
  const std::shared_ptr<detail::JobWorkspace> &before_workspace =
      carrier_state->steps[0u].job->workspace;
  const std::shared_ptr<detail::JobWorkspace> &reduce_workspace =
      carrier_state->steps[1u].job->workspace;
  const std::shared_ptr<detail::JobWorkspace> &after_workspace =
      carrier_state->steps[2u].job->workspace;
  const std::shared_ptr<detail::JobWorkspace> &empty_workspace =
      carrier_state->steps[3u].job->workspace;
  if ((backend == Backend::Cpu &&
       (before_workspace != nullptr || after_workspace != nullptr)) ||
      (backend != Backend::Cpu &&
       (before_workspace == nullptr || reduce_workspace == nullptr ||
        after_workspace == nullptr || before_workspace == reduce_workspace ||
        before_workspace == after_workspace ||
        reduce_workspace == after_workspace ||
        before_workspace->program != carrier_map_state ||
        after_workspace->program != carrier_map_state ||
        !before_workspace->buffers.empty() ||
        !before_workspace->offsets.empty() ||
        !after_workspace->buffers.empty() ||
        !after_workspace->offsets.empty() ||
        before_workspace->arena == nullptr ||
        before_workspace->arena != reduce_workspace->arena ||
        before_workspace->arena != after_workspace->arena)) ||
      empty_workspace != nullptr) {
    return 12;
  }
  const MemoryStats carrier_memory = carrier->memory();
  if (carrier_memory.resident.current > carrier_plan->peak_bytes ||
      carrier_memory.host.current > carrier_plan->peak_bytes ||
      carrier_memory.tile.current > carrier_plan->peak_bytes) {
    return 12;
  }
  std::array<std::int32_t, source_values.size()> before_values{};
  std::array<std::int32_t, source_values.size()> after_values{};
  std::array<std::int32_t, 1u> carrier_reduce_value{};
  constexpr std::array<std::int32_t, source_values.size()> expected_carrier{
      1, 2, 3, 4, 5, 6, 7, 8};
  if (!carrier->run() || !ReadExact(*carrier, *carrier_before, before_values) ||
      !ReadExact(*carrier, *carrier_after, after_values) ||
      !ReadExact(*carrier, *carrier_reduce, carrier_reduce_value) ||
      before_values != expected_carrier || after_values != expected_carrier ||
      carrier_reduce_value != std::array<std::int32_t, 1u>{16}) {
    return 13;
  }

  // Recurrence phases and transactional alternates are logical command
  // owners, not simultaneous dense View storage lifetimes. External input and
  // output Views occur on opposite phases, so every Job borrows one slot sized
  // by maximum phase liveness rather than iteration or owner count.
  auto recurrent_program =
      on(device)
          .input<std::int32_t>(source_values.size())
          .zip_input<std::int32_t>(4u)
          .branch([](auto state, auto values) {
            auto advanced = state.map("pipeline-view-recurrence",
                                      [](auto value) { return value + 1; });
            auto prefix = values.scan(Scan::InclusiveSum);
            return outputs(advanced, prefix);
          })
          .compile();
  auto recurrent_first = Upload(device, source_values);
  auto recurrent_second = device.buffer<std::int32_t>(source_values.size());
  auto recurrent_source = Upload(device, source_values);
  auto recurrent_target = device.buffer<std::int32_t>(source_values.size());
  if (!recurrent_program || !recurrent_first || !recurrent_second ||
      !recurrent_source || !recurrent_target) {
    return 5;
  }
  auto recurrent_input = recurrent_source->view(1u, 4u, 2u);
  auto recurrent_output = recurrent_target->view(1u, 4u, 2u);
  if (!recurrent_input || !recurrent_output) {
    return 5;
  }
  auto recurrent_builder =
      pipeline(device)
          .state(*recurrent_first, *recurrent_second)
          .repeat<6u>(*recurrent_program,
                      read(*recurrent_first, *recurrent_input),
                      write_final(*recurrent_second, *recurrent_output))
          .commit();
  const auto recurrent_plan = recurrent_builder.plan();
  if (!recurrent_plan ||
      recurrent_plan->prepared_buffer_bytes < recurrent_plan->scratch_bytes ||
      recurrent_plan->prepared_bytes !=
          recurrent_plan->prepared_buffer_bytes +
              recurrent_plan->prepared_host_bytes +
              recurrent_plan->prepared_tile_bytes +
              recurrent_plan->prepared_native_bytes) {
    return 6;
  }
  auto recurrent = std::move(recurrent_builder).prepare();
  const std::shared_ptr<detail::PipelineState> recurrent_state =
      recurrent ? detail::PipelineStateAccess::state(*recurrent)
                : std::shared_ptr<detail::PipelineState>{};
  if (recurrent_state == nullptr || recurrent_state->steps.size() != 6u ||
      recurrent_state->prepared_buffers.size() !=
          expected_owners + recurrent_plan->scratch_count) {
    return 7;
  }
  if (backend == Backend::Cpu) {
    std::array<const detail::JobState *, 12u> jobs{};
    std::size_t job_count = 0u;
    std::uint64_t view_bytes = 0u;
    const auto account = [&](const std::shared_ptr<detail::JobState> &job) {
      if (job == nullptr || std::find(jobs.begin(), jobs.begin() + job_count,
                                      job.get()) != jobs.begin() + job_count) {
        return;
      }
      jobs[job_count++] = job.get();
      const auto account_transfers = [&](const auto &transfers,
                                         const auto &owners) {
        for (const detail::CpuViewTransfer &transfer : transfers) {
          if (transfer.binding < owners.size() &&
              owners[transfer.binding] != nullptr) {
            view_bytes += owners[transfer.binding]->bytes;
          }
        }
      };
      account_transfers(job->cpu_view_inputs, job->inputs);
      account_transfers(job->cpu_view_outputs, job->outputs);
    };
    for (const detail::PipelineStep &step : recurrent_state->steps) {
      account(step.job);
      account(step.alternate_job);
    }
    if (view_bytes !=
        recurrent_plan->prepared_buffer_bytes - recurrent_plan->scratch_bytes) {
      return 7;
    }
  } else if (recurrent_plan->prepared_buffer_bytes !=
             4u * sizeof(std::int32_t) + recurrent_plan->scratch_bytes) {
    return 7;
  }
  for (const detail::PipelineStep &step : recurrent_state->steps) {
    if (step.job == nullptr || step.alternate_job == nullptr ||
        !step.job->graph_buffers.empty() ||
        !step.alternate_job->graph_buffers.empty() ||
        (expected_owners == 0u && (step.job->workspace != nullptr ||
                                   step.alternate_job->workspace != nullptr)) ||
        (expected_owners != 0u &&
         (step.job->workspace == nullptr ||
          step.alternate_job->workspace != step.job->workspace ||
          step.job->workspace !=
              recurrent_state->steps.front().job->workspace ||
          step.job->workspace->arena == nullptr ||
          step.job->workspace->arena !=
              recurrent_state->steps.front().job->workspace->arena))) {
      return 8;
    }
  }
  std::array<std::int32_t, 8u> recurrent_values{};
  std::array<std::int32_t, 8u> recurrent_state_values{};
  if (!recurrent || !recurrent->run() || recurrent->generation() != 1u ||
      !ReadExact(*recurrent, *recurrent_target, recurrent_values) ||
      !ReadExact(*recurrent, *recurrent_second, recurrent_state_values) ||
      recurrent_values !=
          std::array<std::int32_t, 8u>{0, 1, 0, 9, 0, 44, 0, 156} ||
      recurrent_state_values !=
          std::array<std::int32_t, 8u>{6, 7, 8, 9, 10, 11, 12, 13}) {
    return 9;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
