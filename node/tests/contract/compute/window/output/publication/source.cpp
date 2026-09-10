#include "../../../pipeline/local.hpp"
#include "../local.hpp"

#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/pipeline/state/assembly.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::test_contract::window {

[[nodiscard]] bool PublicationSourceCoordinates() {
  using namespace rund::compute::detail;
  constexpr std::array<std::size_t, 4u> ordinary_steps{0u, 1u, 2u, 3u};
  constexpr std::array<std::size_t, 4u> nested_routes{0u, 1u, 2u, 1u};
  constexpr std::array<std::uint32_t, 4u> banks{
      PipelineWindow::first,
      PipelineWindow::second,
      PipelineWindow::first,
      PipelineWindow::second,
  };
  for (std::size_t index = 0u; index < ordinary_steps.size(); ++index) {
    const std::size_t bound = index + 1u;
    PipelineBuildState ordinary{};
    ordinary.steps.resize(bound);
    for (std::size_t iteration = 0u; iteration < bound; ++iteration) {
      ordinary.steps[iteration].iteration =
          static_cast<std::uint32_t>(iteration);
      ordinary.steps[iteration].iteration_bound =
          static_cast<std::uint32_t>(bound);
      ordinary.steps[iteration].window_control = {.value = 0u};
    }
    ordinary.window_controls.push_back(PipelineBuildWindowControl{
        .ordinary_step = {.value = 0u},
    });
    const PipelineBuildTerminalPublication publication{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const auto final = resolve_build_window_final(ordinary, {.value = 0u});
    const auto source = resolve_publication_source(ordinary, publication);
    if (!final || !source ||
        final->source_step.value != ordinary_steps[index] ||
        source->step.value != ordinary_steps[index] ||
        final->bank != banks[index]) {
      return false;
    }

    PipelineBuildState nested{};
    const std::size_t fold_first = bound;
    nested.steps.resize(bound + 3u);
    for (std::size_t iteration = 0u; iteration < bound; ++iteration) {
      nested.steps[iteration].iteration = static_cast<std::uint32_t>(iteration);
      nested.steps[iteration].iteration_bound =
          static_cast<std::uint32_t>(bound);
      nested.steps[iteration].window_control = {.value = 0u};
      nested.steps[iteration].nested = 1u;
      nested.steps[iteration].route = PipelineRoute::NestedSeed;
    }
    for (std::size_t route = 0u; route < 3u; ++route) {
      nested.steps[fold_first + route].iteration =
          static_cast<std::uint32_t>(route);
      nested.steps[fold_first + route].iteration_bound = 3u;
      nested.steps[fold_first + route].window_control = {.value = 0u};
      nested.steps[fold_first + route].nested = 1u;
      nested.steps[fold_first + route].route = PipelineRoute::NestedFold;
    }
    nested.window_controls.push_back(PipelineBuildWindowControl{
        .nested = 1u,
    });
    node::accel::detail::NestedTemplateShape nested_shape{};
    if (!node::accel::detail::ProveNestedTemplateShape(
            0u, static_cast<std::uint32_t>(bound), 1u, 0u, nested_shape)) {
      return false;
    }
    nested.nested_windows.push_back(PipelineBuildNestedWindow{
        .shape = nested_shape,
    });
    const PipelineBuildTerminalPublication terminal{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const PipelineBuildWindowPublication window{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const auto nested_final = resolve_build_window_final(nested, {.value = 0u});
    const auto terminal_source = resolve_publication_source(nested, terminal);
    const auto window_source = resolve_publication_source(nested, window);
    if (!nested_final || !terminal_source || !window_source ||
        nested_final->source_step.value != fold_first + nested_routes[index] ||
        terminal_source->step.value != fold_first + nested_routes[index] ||
        window_source->step.value != fold_first ||
        nested_final->bank != banks[index]) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::test_contract::window
