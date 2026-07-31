#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildGatherBinds(const KernelExecutionStep &step,
                      const RunBinds &run_binds, GatherBinds &out) {
  const auto &active = step.operation.get<operation::Gather>();
  const bool bounded =
      active.plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != (bounded ? 4u : 3u)) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  std::size_t index = 0u;
  if (!ReadBinding(source, step.graph_binding_indices[index++], out.values,
                   out.values_handle) ||
      !ReadBinding(source, step.graph_binding_indices[index++], out.indices,
                   out.indices_handle)) {
    return false;
  }
  if (bounded && !ReadBinding(source, step.graph_binding_indices[index++],
                              out.logical_count, out.logical_count_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[index], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.values_handle) && BindingReady(out.indices_handle) &&
         (!bounded || BindingReady(out.logical_count_handle)) &&
         BindingReady(out.output_handle);
}

} // namespace rund::node::accel::detail
