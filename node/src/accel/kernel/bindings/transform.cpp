#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildTransformBinds(const KernelExecutionStep& step,
                         const RunBinds& run_binds,
                         TransformBinds& out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 4u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input_real,
                   out.input_real_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.input_imag,
                   out.input_imag_handle) ||
      !ReadBinding(source, step.graph_binding_indices[2u], out.output_real,
                   out.output_real_handle) ||
      !ReadBinding(source, step.graph_binding_indices[3u], out.output_imag,
                   out.output_imag_handle)) {
    return false;
  }
  return BindingReady(out.input_real_handle) &&
         BindingReady(out.input_imag_handle) &&
         BindingReady(out.output_real_handle) &&
         BindingReady(out.output_imag_handle);
}

}  // namespace rund::node::accel::detail
