#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildMatrixBinds(const KernelExecutionStep &step,
                      const RunBinds &run_binds, MatrixBinds &out) {
  const auto &active = step.operation.get<operation::Matrix>();
  const bool transpose = active.desc.op == rund::kernel::MatrixOp::Transpose;
  const std::uint64_t expected = transpose ? 2u : 3u;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != expected) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.left,
                   out.left_handle)) {
    return false;
  }
  if (transpose) {
    if (!ReadBinding(source, step.graph_binding_indices[1u], out.output,
                     out.output_handle)) {
      return false;
    }
    return BindingReady(out.left_handle) && BindingReady(out.output_handle);
  }
  if (!ReadBinding(source, step.graph_binding_indices[1u], out.right,
                   out.right_handle) ||
      !ReadBinding(source, step.graph_binding_indices[2u], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.left_handle) && BindingReady(out.right_handle) &&
         BindingReady(out.output_handle);
}

} // namespace rund::node::accel::detail
