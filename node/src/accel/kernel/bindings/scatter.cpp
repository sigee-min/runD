#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildScatterBinds(const KernelExecutionStep &step,
                       const RunBinds &run_binds, ScatterBinds &out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 3u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.values,
                   out.values_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.indices,
                   out.indices_handle) ||
      !ReadBinding(source, step.graph_binding_indices[2u], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.values_handle) && BindingReady(out.indices_handle) &&
         BindingReady(out.output_handle);
}

bool BuildScatterReduceBinds(const KernelExecutionStep &step,
                             const RunBinds &run_binds,
                             ScatterReduceBinds &out) {
  const auto &active = step.operation.get<operation::ScatterReduce>();
  const bool bounded =
      active.desc.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const std::size_t count = bounded ? 4u : 3u;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != count)
    return false;
  const BindingSource source = BindingSourceFor(run_binds);
  const std::size_t output = count - 1u;
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.values,
                   out.values_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.indices,
                   out.indices_handle) ||
      !ReadBinding(source, step.graph_binding_indices[output], out.output,
                   out.output_handle) ||
      (bounded && !ReadBinding(source, step.graph_binding_indices[2u],
                               out.count, out.count_handle)))
    return false;
  return BindingReady(out.values_handle) && BindingReady(out.indices_handle) &&
         BindingReady(out.output_handle) &&
         (!bounded || BindingReady(out.count_handle));
}

} // namespace rund::node::accel::detail
