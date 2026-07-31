#include "../local.hpp"

#include "../../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildSegmentedReduceBinds(const KernelExecutionStep &step,
                               const RunBinds &run_binds,
                               SegmentedReduceBinds &out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 3u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input,
                   out.input_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.heads,
                   out.heads_handle) ||
      !ReadBinding(source, step.graph_binding_indices[2u], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.input_handle) && BindingReady(out.heads_handle) &&
         BindingReady(out.output_handle);
}

} // namespace rund::node::accel::detail
