#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildSolveBinds(const KernelExecutionStep &step, const RunBinds &run_binds,
                     SolveBinds &out) {
  const auto &active = step.operation.get<operation::Solve>();
  const bool factor_input =
      active.desc.input == rund::kernel::SolveInput::Factor;
  const bool needs_aux =
      factor_input && active.desc.factor == rund::kernel::FactorOp::LU;
  const std::uint64_t expected = factor_input ? (needs_aux ? 5u : 4u) : 4u;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != expected) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.primary,
                   out.primary_handle)) {
    return false;
  }
  const std::uint64_t rhs_index = factor_input ? (needs_aux ? 2u : 1u) : 1u;
  if (needs_aux && !ReadBinding(source, step.graph_binding_indices[1u], out.aux,
                                out.aux_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[rhs_index], out.rhs,
                   out.rhs_handle) ||
      !ReadBinding(source, step.graph_binding_indices[rhs_index + 1u],
                   out.output, out.output_handle) ||
      !ReadBinding(source, step.graph_binding_indices[rhs_index + 2u],
                   out.status, out.status_handle)) {
    return false;
  }
  return BindingReady(out.primary_handle) && BindingReady(out.rhs_handle) &&
         BindingReady(out.output_handle) && BindingReady(out.status_handle) &&
         (!needs_aux || BindingReady(out.aux_handle));
}

} // namespace rund::node::accel::detail
