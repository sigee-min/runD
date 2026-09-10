#include "../graph.hpp"
#include "../state/program.hpp"
#include "primitive/algebra.hpp"
#include "primitive/local.hpp"
#include "state.hpp"

#include <array>
#include <span>
#include <variant>

namespace rund::compute::detail {

namespace {

// This coordinator owns the one binding pass and the operation-family route.
// Family implementations consume the same borrowed ports and CpuGraphRun;
// they do not create a second execution or scratch owner.
[[nodiscard]] Status run_cpu_primitive(JobState &job, CpuGraphProgram &cpu,
                                       CpuGraphRun &run,
                                       const std::size_t step_index,
                                       const CpuRuntimePrimitive &primitive) {
  constexpr std::size_t kMaxPorts = 8u;
  std::array<RawCpuBuffer, kMaxPorts> ports{};
  const Status bound = bind_cpu_primitive_ports(job, cpu, run, step_index,
                                                std::span<RawCpuBuffer>{ports});
  if (!bound) {
    return bound;
  }
  const std::size_t count = cpu.bind_count[step_index];
  PrimitiveContext context{
      job,       cpu,
      run,       step_index,
      primitive, std::span<const RawCpuBuffer>{ports.data(), count}};
  switch (primitive.kind) {
  case Primitive::SegmentedScan:
  case Primitive::SegmentedReduce:
    return run_cpu_primitive_collective(context);
  case Primitive::Sort:
  case Primitive::Argsort:
    return run_cpu_primitive_ordering(context);
  case Primitive::Compact:
  case Primitive::Histogram:
  case Primitive::Partition:
    return run_cpu_primitive_reference(context);
  case Primitive::Gather:
    return run_cpu_primitive_indexed(context);
  case Primitive::Scatter:
    return run_cpu_primitive_ordering(context);
  case Primitive::ScatterReduce:
    return run_cpu_primitive_indexed(context);
  case Primitive::Stencil:
    return run_stencil(context);
  case Primitive::Window:
    return run_window(context);
  case Primitive::Transform:
    return run_transform(context);
  case Primitive::Matrix:
    return run_matrix(context);
  case Primitive::Factor:
    return run_factor(context);
  case Primitive::Solve:
    return run_solve(context);
  case Primitive::Spectrum:
    return run_spectrum(context);
  case Primitive::Reduce:
    return Status::fail(Reason::ReduceRouteInvalid);
  }
  return finish_cpu_primitive(false, "compute_primitive_unsupported");
}

} // namespace

Status execute_cpu_primitive(JobState &job, const std::size_t step) {
  if (job.program == nullptr || job.program->cpu_graph == nullptr ||
      job.cpu == nullptr || job.cpu->graph == nullptr ||
      job.cpu->graph->storage == nullptr || job.inputs.empty() ||
      job.outputs.empty()) {
    return Status::fail(Reason::RunInvalid);
  }
  CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &graph = *program.runtime;
  if (step >= graph.steps.size()) {
    return Status::fail(Reason::GraphStepInvalid);
  }
  const auto *const primitive =
      std::get_if<CpuRuntimePrimitive>(&graph.steps[step]);
  if (primitive == nullptr || primitive->kind == Primitive::Reduce) {
    return Status::fail(Reason::PrimitiveRouteInvalid);
  }
  return run_cpu_primitive(job, program, *job.cpu->graph, step, *primitive);
}

} // namespace rund::compute::detail
