#include "local.hpp"

#include "../../context/internal.hpp"

namespace rund::node::accel::detail {

bool BuildHistogramBinds(const KernelExecutionStep& step,
                         const RunBinds& run_binds,
                         HistogramBinds& out) {
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != 2u) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  return ReadBinding(source, step.graph_binding_indices[0u], out.bins,
                     out.bins_handle) &&
         ReadBinding(source, step.graph_binding_indices[1u], out.counts,
                     out.counts_handle) &&
         BindingReady(out.bins_handle) && BindingReady(out.counts_handle);
}

}  // namespace rund::node::accel::detail
