#include "../../allocation.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
template <class Seed, class Fold>
[[nodiscard]] int CheckSealedPublicationMutation(Device &device,
                                                 const Seed &seed,
                                                 const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 3u> count_values{
      0x13579BDFu, static_cast<std::uint32_t>(kOutputMaximum), 0x2468ACE0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputWindowBacking> window_values{};
  window_values.fill(kOutputSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended) {
    return 1;
  }
  auto count_view = count->view(1u, 1u);
  auto appended_view = appended->view(kOutputWindowOffset, kOutputMaximum);
  if (!count_view || !appended_view) {
    return 2;
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
      BufferAccess::view(*appended_view, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count_view, ResourceAccess::Read), inputs, finals,
      windows, kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
  for (std::size_t route = 1u; route < 3u; ++route) {
    auto drift = make_pipeline(DeviceAccess::state(device));
    append_pipeline_window_repeat(
        drift, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
        BufferAccess::view(*count_view, ResourceAccess::Read), inputs, finals,
        windows, kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
    if (drift == nullptr || drift->failure != Reason::Ok ||
        drift->nested_windows.size() != 1u) {
      return 3;
    }
    const PipelineBuildNestedWindow &nested = drift->nested_windows[0u];
    PipelineBinding &route_output =
        drift->steps[nested.shape.fold_first() + route]
            .outputs[nested.recurrent_output_count];
    route_output.owner = static_cast<std::uint32_t>(drift->internals.size());
    drift->internals.push_back(PipelineInternal{
        .type = route_output.type,
        .format = route_output.format,
        .count = route_output.count,
    });
    const auto invalid = plan_pipeline(drift);
    if (invalid || invalid.reason() != Reason::PipelineInvalid) {
      return static_cast<int>(9u + route);
    }
  }

  const auto plan = plan_pipeline(build);
  if (!plan || build == nullptr || build->memory == nullptr ||
      build->memory->publications.size() != 2u) {
    return 3;
  }
  auto authored = std::find_if(
      build->publications.begin(), build->publications.end(),
      [](const PipelineBuildPublication &publication) {
        return std::holds_alternative<PipelineBuildWindowPublication>(
            publication);
      });
  auto authored_terminal = std::find_if(
      build->publications.begin(), build->publications.end(),
      [](const PipelineBuildPublication &publication) {
        return std::holds_alternative<PipelineBuildTerminalPublication>(
            publication);
      });
  auto sealed = std::find_if(
      build->memory->publications.begin(), build->memory->publications.end(),
      [](const PipelinePublicationPlan &publication) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(
            publication);
      });
  if (authored == build->publications.end() ||
      authored_terminal == build->publications.end() ||
      sealed == build->memory->publications.end()) {
    return 4;
  }
  auto &authored_window = std::get<PipelineBuildWindowPublication>(*authored);
  const auto &sealed_window = std::get<PipelineWindowPublicationPlan>(*sealed);
  if (build->nested_windows.size() != 1u ||
      build->memory->window_controls.size() != 1u) {
    return 5;
  }
  const PipelineWindowControl &sealed_control =
      build->memory->window_controls[0u];
  const std::uint64_t sealed_offset =
      sealed_window.target.view.identity.offset_bytes;
  if (sealed_offset != kOutputWindowOffset * sizeof(std::uint32_t) ||
      sealed_control.count.identity.offset_bytes != sizeof(std::uint32_t) ||
      sealed_control.maximum != kOutputMaximum ||
      sealed_control.tile != kOutputTile) {
    return 5;
  }

  // Mutate the cold authored publication and the sole authored control after
  // planning without invalidating memory. Preparation, accounting,
  // fingerprinting, and execution must retain the already sealed authority.
  authored_window.edge.target.offset = kOutputWindowOffset + 1u;
  authored_window.edge.control = {};
  authored_window.edge.output.value = 0u;
  auto &authored_final =
      std::get<PipelineBuildTerminalPublication>(*authored_terminal);
  authored_final.edge.target.offset = 1u;
  authored_final.edge.control = {};
  authored_final.edge.output.value = 1u;
  if (build->window_controls.size() != 1u) {
    return 6;
  }
  PipelineBuildWindowControl &authored_control = build->window_controls[0u];
  authored_control.count_input = 0u;
  node::accel::detail::NestedTemplateShape drifted_shape{};
  if (!node::accel::detail::ProveNestedTemplateShape(
          build->nested_windows[0u].shape.first() + 1u,
          authored_control.maximum, authored_control.tile,
          build->nested_windows[0u].shape.inner_bound(), drifted_shape)) {
    return 6;
  }
  build->nested_windows[0u].shape = drifted_shape;
  authored_control.maximum = kOutputTile;
  authored_control.tile = 1u;
  authored_control.terminal = 0u;
  authored_control.expected = 0xDEADBEEFu;
  auto prepared = prepare_pipeline(build);
  if (!prepared || (*prepared)->publications.size() != 2u) {
    return 6;
  }
  // Lock the complete v3 Pipeline identity, not only the isolated publication
  // suffix. In particular, nested routes retain zero/default ordinary-window
  // slots and serialize their control once in the nested-begin block.
  constexpr Fingerprint expected_nested{
      .hi = 0x99db298093fd0b2full,
      .lo = 0x8535483e8b5c9539ull,
  };
  if ((*prepared)->publication->fingerprint != expected_nested) {
    return 9;
  }
  const auto runtime = std::find_if(
      (*prepared)->publications.begin(), (*prepared)->publications.end(),
      [](const PipelinePublicationPlan &publication) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(
            publication);
      });
  if (runtime == (*prepared)->publications.end() ||
      std::get<PipelineWindowPublicationPlan>(*runtime)
              .target.view.identity.offset_bytes != sealed_offset ||
      !run_pipeline(*prepared)) {
    return 7;
  }

  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kOutputWindowBacking> actual_window{};
  const Status final_read = read_pipeline_raw(
      *prepared, BufferAccess::state(*final), Type::U32, FixedFormat{},
      actual_final.data(), sizeof(actual_final), actual_final.size());
  const Status window_read = read_pipeline_raw(
      *prepared, BufferAccess::state(*appended), Type::U32, FixedFormat{},
      actual_window.data(), sizeof(actual_window), actual_window.size());
  std::array<std::uint32_t, kOutputWindowBacking> expected_window{};
  expected_window.fill(kOutputSentinel);
  const auto expected_payload = ExpectedWindow(kOutputValues, kOutputMaximum);
  std::copy(expected_payload.begin(), expected_payload.end(),
            expected_window.begin() + kOutputWindowOffset);
  return final_read && window_read &&
                 actual_final[0u] ==
                     ExpectedFinal(kOutputValues, kOutputMaximum) &&
                 actual_window == expected_window
             ? 0
             : 8;
}

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

int CheckOutputSealedPublicationMutation(Device &device, const OutputSeed &seed,
                                         const OutputFold &fold) {
  return CheckSealedPublicationMutation(device, seed, fold);
}
int CheckOutputPublicationJobBindingMutation(Device &device,
                                             const OutputSeed &seed,
                                             const OutputFold &fold) {
  return CheckPublicationJobBindingMutation(device, seed, fold);
}

} // namespace rund::node::test_contract::window
