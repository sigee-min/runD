#include "window/internal.hpp"

#include <rund/compute/pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>

namespace rund::compute::detail {

void append_pipeline_window_repeat(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &seed,
    const std::shared_ptr<ProgramState> &action,
    const std::shared_ptr<ProgramState> &fold, const ResourceView &resident,
    const std::span<const ResourceView> inputs,
    const std::span<const ResourceView> final_outputs,
    const std::span<const ResourceView> window_outputs,
    const std::size_t maximum, const std::size_t tile, const std::size_t inner,
    const std::size_t terminal, const std::uint32_t expected) noexcept {
  if (build == nullptr || build->failure != Reason::Ok) {
    return;
  }

  const WindowAssemblyInput input{
      .seed = &seed,
      .action = &action,
      .fold = &fold,
      .resident = &resident,
      .inputs = inputs,
      .final_outputs = final_outputs,
      .window_outputs = window_outputs,
      .maximum = maximum,
      .tile = tile,
      .inner = inner,
      .terminal = terminal,
      .expected = expected,
  };
  WindowAssemblyCounts counts{};
  if (!validate_window_request(*build, input, counts)) {
    return;
  }

  // Construction follows the complete non-mutating proof.  From this point
  // through publication every build vector is covered by one rollback owner.
  PipelineBuildMutation mutation{*build};
  try {
    WindowAssemblyResources resources{};
    if (!materialize_window_resources(*build, input, counts, resources,
                                      mutation) ||
        !emit_window_steps(*build, input, counts, resources, mutation) ||
        !publish_window(*build, input, counts, resources)) {
      return;
    }
    mutation.commit();
  } catch (const std::bad_alloc &) {
    // Allocation failure is the only exception class this noexcept builder
    // projects. Unexpected exceptions retain the existing terminate contract.
    mutation.fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
