#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildSpectrumBinds(const KernelExecutionStep &step,
                        const RunBinds &run_binds, SpectrumBinds &out) {
  const auto &active = step.operation.get<operation::Spectrum>();
  const bool has_vectors = active.plan.vector_count != 0u;
  const std::uint64_t expected = has_vectors ? 4u : 3u;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != expected) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input,
                   out.input_handle) ||
      !ReadBinding(source, step.graph_binding_indices[1u], out.values,
                   out.values_handle)) {
    return false;
  }
  const std::uint64_t status_index = has_vectors ? 3u : 2u;
  if (has_vectors && !ReadBinding(source, step.graph_binding_indices[2u],
                                  out.vectors, out.vectors_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[status_index], out.status,
                   out.status_handle)) {
    return false;
  }
  return BindingReady(out.input_handle) && BindingReady(out.values_handle) &&
         BindingReady(out.status_handle) &&
         (!has_vectors || BindingReady(out.vectors_handle));
}

} // namespace rund::node::accel::detail
