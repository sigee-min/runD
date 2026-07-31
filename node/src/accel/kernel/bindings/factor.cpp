#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildFactorBinds(const KernelExecutionStep &step,
                      const RunBinds &run_binds, FactorBinds &out) {
  const auto &active = step.operation.get<operation::Factor>();
  const bool needs_aux = active.desc.op == rund::kernel::FactorOp::LU;
  const std::uint64_t expected = needs_aux ? 4u : 3u;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != expected) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input,
                   out.input_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.factor,
                   out.factor_handle)) {
    return false;
  }
  const std::uint64_t status_index = needs_aux ? 3u : 2u;
  if (needs_aux && !ReadBinding(source, step.graph_binding_indices[2u], out.aux,
                                out.aux_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[status_index], out.status,
                   out.status_handle)) {
    return false;
  }
  return BindingReady(out.input_handle) && BindingReady(out.factor_handle) &&
         BindingReady(out.status_handle) &&
         (!needs_aux || BindingReady(out.aux_handle));
}

} // namespace rund::node::accel::detail
