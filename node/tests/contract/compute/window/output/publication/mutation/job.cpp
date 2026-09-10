#include "contract/compute/allocation.hpp"
#include "contract/compute/pipeline/local.hpp"
#include "contract/compute/window/local.hpp"
#include "contract/compute/window/output/local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/pipeline/state/assembly.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {

template <class Seed, class Fold>
[[nodiscard]] int CheckPublicationJobBindingMutation(Device &device,
                                                     const Seed &seed,
                                                     const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(kOutputMaximum)};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputMaximum> window_values{};
  window_values.fill(kOutputSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  auto second_count = device.upload<std::uint32_t>(count_values);
  auto ordinary_final_first = device.buffer<std::uint32_t>(1u);
  auto ordinary_tile_first = device.buffer<std::uint32_t>(kOutputTile);
  auto ordinary_final_second = device.buffer<std::uint32_t>(1u);
  auto ordinary_tile_second = device.buffer<std::uint32_t>(kOutputTile);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !second_count || !ordinary_final_first ||
      !ordinary_tile_first || !ordinary_final_second || !ordinary_tile_second) {
    return 1;
  }

  const std::array<ResourceView, 4u> inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*values, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> finals{
      BufferAccess::view(*final, ResourceAccess::Write),
  };
  const std::array<ResourceView, 1u> windows{
      BufferAccess::view(*appended, ResourceAccess::Write),
  };
  const std::array<ResourceView, 3u> ordinary_inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 2u> ordinary_outputs_first{
      BufferAccess::view(*ordinary_final_first, ResourceAccess::Write),
      BufferAccess::view(*ordinary_tile_first, ResourceAccess::Write),
  };
  const std::array<ResourceView, 2u> ordinary_outputs_second{
      BufferAccess::view(*ordinary_final_second, ResourceAccess::Write),
      BufferAccess::view(*ordinary_tile_second, ResourceAccess::Write),
  };
  const auto make_build = [&] {
    auto build = make_pipeline(DeviceAccess::state(device));
    append_pipeline_window_repeat(
        build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
        BufferAccess::view(*count, ResourceAccess::Read), inputs, finals,
        windows, kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
    return build;
  };

  auto bank_build = make_build();
  const auto bank_plan = plan_pipeline(bank_build);
  if (!bank_plan || bank_build == nullptr || bank_build->memory == nullptr ||
      bank_build->nested_windows.size() != 1u) {
    return 2;
  }
  const std::size_t fold_first =
      bank_build->nested_windows[0u].shape.fold_first();
  if (fold_first > bank_build->steps.size() ||
      bank_build->steps.size() - fold_first < 3u ||
      bank_build->steps[fold_first + 1u].outputs.empty() ||
      bank_build->steps[fold_first + 2u].outputs.empty()) {
    return 3;
  }
  const std::size_t seed_first =
      bank_build->nested_windows[0u].shape.seed_first();
  if (seed_first >= bank_build->steps.size() ||
      bank_build->steps[seed_first].inputs.size() < 2u) {
    return 4;
  }
  // The cached plan says route 2 writes the first recurrent bank. Redirect the
  // authored mirror to an already admitted owner without invalidating memory;
  // Job materialization must still consume the frozen step binding.
  PipelineBinding redirected =
      bank_build->steps[seed_first]
          .inputs[bank_build->steps[seed_first].inputs.size() - 2u];
  redirected.access = ResourceAccess::Write;
  redirected.hidden = true;
  bank_build->steps[fold_first + 2u].outputs[0u] = std::move(redirected);
  auto bank_prepared = prepare_pipeline(bank_build);
  if (!bank_prepared) {
    return 5;
  }

  auto count_build = make_build();
  const auto count_plan = plan_pipeline(count_build);
  if (!count_plan || count_build == nullptr || count_build->memory == nullptr ||
      count_build->nested_windows.size() != 1u) {
    return 6;
  }
  const PipelineBuildNestedWindow &nested = count_build->nested_windows[0u];
  if (nested.shape.seed_count() < 2u ||
      nested.shape.seed_first() >= count_build->steps.size() ||
      nested.shape.seed_first() + 1u >= count_build->steps.size()) {
    return 7;
  }
  PipelineBuildStep &later_seed =
      count_build->steps[nested.shape.seed_first() + 1u];
  if (later_seed.inputs.size() < 3u) {
    return 8;
  }
  // Replace the authored penultimate count input with the already admitted
  // one-element witness View. Every Seed Job must still use the frozen count
  // coordinate rather than this stale declaration.
  later_seed.inputs[later_seed.inputs.size() - 2u] = later_seed.inputs[2u];
  auto count_prepared = prepare_pipeline(count_build);
  if (!count_prepared) {
    return 9;
  }

  auto drift_build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(drift_build, ProgramAccess::state(fold),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          ordinary_inputs, ordinary_outputs_first,
                          kOutputMaximum, kOutputTile, NoWindowTerminal, 1u);
  append_pipeline_windows(
      drift_build, ProgramAccess::state(fold),
      BufferAccess::view(*second_count, ResourceAccess::Read), ordinary_inputs,
      ordinary_outputs_second, kOutputMaximum, kOutputTile, NoWindowTerminal,
      1u);
  if (drift_build == nullptr || drift_build->failure != Reason::Ok ||
      drift_build->window_controls.size() != 2u) {
    return 10;
  }
  const std::size_t drift_step =
      drift_build->window_controls[0u].ordinary_step.value + 1u;
  if (drift_step >= drift_build->steps.size()) {
    return 11;
  }
  drift_build->steps[drift_step].window_control = {.value = 1u};
  const auto drift_plan = plan_pipeline(drift_build);
  if (drift_plan || drift_plan.reason() != Reason::PipelineInvalid) {
    return 12;
  }
  drift_build->steps[drift_step].window_control = {};
  const auto unassigned_plan = plan_pipeline(drift_build);
  if (unassigned_plan || unassigned_plan.reason() != Reason::PipelineInvalid) {
    return 13;
  }

  for (const bool nested_first : {false, true}) {
    auto mixed_build = make_pipeline(DeviceAccess::state(device));
    const auto append_ordinary = [&] {
      append_pipeline_windows(
          mixed_build, ProgramAccess::state(fold),
          BufferAccess::view(*second_count, ResourceAccess::Read),
          ordinary_inputs, ordinary_outputs_second, kOutputMaximum, kOutputTile,
          NoWindowTerminal, 1u);
    };
    const auto append_nested = [&] {
      append_pipeline_window_repeat(
          mixed_build, ProgramAccess::state(seed), {},
          ProgramAccess::state(fold),
          BufferAccess::view(*count, ResourceAccess::Read), inputs, finals,
          windows, kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
    };
    if (nested_first) {
      append_nested();
      append_ordinary();
    } else {
      append_ordinary();
      append_nested();
    }
    const auto mixed_plan = plan_pipeline(mixed_build);
    if (!mixed_plan || mixed_build == nullptr ||
        mixed_build->memory == nullptr ||
        mixed_build->memory->window_controls.size() != 2u) {
      return 14;
    }
    auto mixed_prepared = prepare_pipeline(mixed_build);
    if (!mixed_prepared || !run_pipeline(*mixed_prepared)) {
      return 15;
    }
  }

  auto coordinate_build = make_build();
  const auto coordinate_plan = plan_pipeline(coordinate_build);
  if (!coordinate_plan || coordinate_build == nullptr ||
      coordinate_build->memory == nullptr ||
      coordinate_build->memory->window_controls.size() != 1u) {
    return 16;
  }
  auto mutable_plan =
      std::const_pointer_cast<PipelineMemoryPlan>(coordinate_build->memory);
  mutable_plan->window_controls[0u].count_input = 0u;
  auto coordinate_prepared = prepare_pipeline(coordinate_build);
  if (coordinate_prepared ||
      coordinate_prepared.reason() != Reason::PipelineInvalid) {
    return 17;
  }
  const auto verify = [&](const std::shared_ptr<PipelineState> &prepared) {
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kOutputMaximum> actual_window{};
    const auto expected_window = ExpectedWindow(kOutputValues, kOutputMaximum);
    return run_pipeline(prepared) &&
           read_pipeline_raw(prepared, BufferAccess::state(*final), Type::U32,
                             FixedFormat{}, actual_final.data(),
                             sizeof(actual_final), actual_final.size()) &&
           read_pipeline_raw(prepared, BufferAccess::state(*appended),
                             Type::U32, FixedFormat{}, actual_window.data(),
                             sizeof(actual_window), actual_window.size()) &&
           actual_final[0u] == ExpectedFinal(kOutputValues, kOutputMaximum) &&
           actual_window == expected_window;
  };
  return verify(*bank_prepared) && verify(*count_prepared) ? 0 : 18;
}

int CheckOutputPublicationJobBindingMutation(Device &device,
                                             const OutputSeed &seed,
                                             const OutputFold &fold) {
  return CheckPublicationJobBindingMutation(device, seed, fold);
}

} // namespace rund::node::test_contract::window
