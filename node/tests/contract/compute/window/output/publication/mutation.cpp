#include "../../../allocation.hpp"
#include "../../../pipeline/local.hpp"
#include "../../local.hpp"
#include "../local.hpp"

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

int CheckOutputSealedPublicationMutation(Device &device, const OutputSeed &seed,
                                         const OutputFold &fold) {
  return CheckSealedPublicationMutation(device, seed, fold);
}

} // namespace rund::node::test_contract::window
