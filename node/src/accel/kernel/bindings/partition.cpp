#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildPartitionBinds(const KernelExecutionStep& step,
                         const RunBinds& run_binds,
                         PartitionBinds& out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 3u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.flags,
                   out.flags_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.values,
                   out.values_handle) ||
      !ReadBinding(source, step.graph_binding_indices[2u], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.flags_handle) && BindingReady(out.values_handle) &&
         BindingReady(out.output_handle);
}

}  // namespace rund::node::accel::detail
