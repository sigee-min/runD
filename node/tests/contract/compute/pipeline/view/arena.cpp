#include "../local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <memory>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckViewArena(rund::compute::Device &device,
                                 const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  const std::size_t expected_owners = backend == Backend::Cpu ? 0u : 1u;
  const std::uint64_t expected_bytes =
      backend == Backend::Cpu ? 0u : 4u * sizeof(std::int32_t);

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
      mixed_plan->prepared_bytes != expected_bytes + mixed_plan->scratch_bytes) {
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
      recurrent_plan->prepared_bytes !=
          expected_bytes + recurrent_plan->scratch_bytes) {
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
  for (const detail::PipelineStep &step : recurrent_state->steps) {
    if (step.job == nullptr || step.alternate_job == nullptr ||
        step.job->workspace == nullptr ||
        step.alternate_job->workspace != step.job->workspace ||
        step.job->workspace != recurrent_state->steps.front().job->workspace ||
        (expected_owners != 0u &&
         (step.job->workspace->arena == nullptr ||
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
