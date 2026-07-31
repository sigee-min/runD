#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildStencilBinds(const KernelExecutionStep& step,
                       const RunBinds& run_binds,
                       StencilBinds& out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 2u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input,
                   out.input_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.input_handle) && BindingReady(out.output_handle);
}

}  // namespace rund::node::accel::detail
