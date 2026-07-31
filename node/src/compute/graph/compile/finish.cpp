#include "local.hpp"

#include "../../backend.hpp"
#include "../../cpu/graph.hpp"
#include "../../fixed/format.hpp"
#include "../../type.hpp"

#include <kernel/program/compute/graph/validation.hpp>

#include <bit>
#include <new>
#include <utility>

namespace rund::compute::detail::graph_compile {

Result<std::shared_ptr<ProgramState>> finish(Lowering &&lowering) {
  try {
    const auto &identity = lowering.graph->identity_outputs.empty()
                               ? lowering.graph->outputs
                               : lowering.graph->identity_outputs;
    lowering.outputs.assign(identity.begin(), identity.end());
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramCapacity);
  }
  const Type root =
      lowering.graph->values[lowering.graph->inputs.front() - 1u].type;
  const FixedFormat format =
      lowering.graph->values[lowering.graph->inputs.front() - 1u].fixed_format;
  const kernel::ComputeScalar scalar = type_scalar(root);
  if (lowering.cpu()) {
    if (lowering.program->cpu_graph->runtime == nullptr ||
        lowering.program->cpu_graph->runtime->steps.size() !=
            lowering.graph->steps.size()) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::CpuProgramInvalid);
    }
    const kernel::GraphCheck check = kernel::ValidateGraph(kernel::Graph{
        .nodes = lowering.cpu_nodes.data(),
        .node_count = lowering.cpu_nodes.size(),
        .outputs = lowering.outputs.data(),
        .output_count = lowering.outputs.size(),
        .scalar = scalar,
        .domain = type_domain(root),
        .fixed_format = kernel_format(format),
    });
    if (!check.ok) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          project_reason(check.reason, Reason::GraphInvalid));
    }
    if (!lowering.runtime) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          lowering.runtime.reason());
    }
    lowering.program->cpu_graph->graph_hash =
        check.graph_id_hi ^ std::rotl(check.graph_id_lo, 1);
    return Result<std::shared_ptr<ProgramState>>::success(
        std::move(lowering.program));
  }

  try {
    lowering.program->accel = std::make_unique<AccelProgram>();
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramCapacity);
  }
  if (lowering.graph->device->ops == nullptr ||
      lowering.graph->device->ops->compile == nullptr ||
      accel_device(*lowering.graph->device) == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::AccelProgramInvalid);
  }
  const Status compiled = lowering.graph->device->ops->compile(
      *lowering.graph->device, *lowering.program->accel,
      rund::AccelGraph{
          .nodes = lowering.accel_nodes.data(),
          .node_count = lowering.accel_nodes.size(),
          .outputs = lowering.outputs.data(),
          .output_count = lowering.outputs.size(),
          .scalar = scalar,
          .domain = type_domain(root),
          .fixed_format = kernel_format(format),
      });
  return compiled
             ? Result<std::shared_ptr<ProgramState>>::success(
                   std::move(lowering.program))
             : Result<std::shared_ptr<ProgramState>>::fail(compiled.reason());
}

} // namespace rund::compute::detail::graph_compile
