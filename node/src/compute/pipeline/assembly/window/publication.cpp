#include "internal.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

bool publish_window(PipelineBuildState &build, const WindowAssemblyInput &input,
                    const WindowAssemblyCounts &counts,
                    const WindowAssemblyResources &resources) {
  for (std::size_t index = 0u; index < counts.recurrent_count; ++index) {
    build.publications.push_back(PipelineBuildTerminalPublication{
        .edge =
            {
                .target = resources.final[index],
                .control = resources.window_control,
                .output = {.value = static_cast<std::uint32_t>(index)},
            },
    });
  }
  for (std::size_t index = 0u; index < counts.window_count; ++index) {
    build.publications.push_back(PipelineBuildWindowPublication{
        .edge =
            {
                .target = resources.window_target[index],
                .control = resources.window_control,
                .output = {.value = static_cast<std::uint32_t>(
                               counts.recurrent_count + index)},
            },
    });
  }
  build.window_controls.push_back(PipelineBuildWindowControl{
      .count_input = counts.seed_external_count,
      .maximum = input.maximum,
      .tile = input.tile,
      .terminal = input.terminal,
      .expected = input.expected,
      .nested = resources.nested,
  });
  build.nested_windows.push_back(PipelineBuildNestedWindow{
      .shape = counts.nested_shape,
      .recurrent_output_count = counts.recurrent_count,
  });
  build.binding_count += counts.binding_count;
  ++build.logical_step_count;
  changed(build);
  return true;
}

} // namespace rund::compute::detail
