#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildCompactBinds(const KernelExecutionStep &step,
                       const RunBinds &run_binds, CompactBinds &out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 2u) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  return ReadBindingPair(source, step.graph_binding_indices[0u], out.flags,
                         out.flags_handle, step.graph_binding_indices[1u],
                         out.output, out.output_handle) &&
         BindingReady(out.flags_handle) && BindingReady(out.output_handle);
}

} // namespace rund::node::accel::detail
