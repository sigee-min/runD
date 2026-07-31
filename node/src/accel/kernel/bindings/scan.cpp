#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildScanBinds(const KernelExecutionStep &step, const RunBinds &run_binds,
                    ScanBinds &out) {
  const auto &active = step.operation.get<operation::Scan>();
  const bool bounded =
      active.desc.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != (bounded ? 3u : 2u)) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  if (!ReadBinding(source, step.graph_binding_indices[0u], out.input,
                   out.input_handle)) {
    return false;
  }
  const std::size_t output = bounded ? 2u : 1u;
  if (bounded && !ReadBinding(source, step.graph_binding_indices[1u],
                              out.logical_count, out.logical_count_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[output], out.output,
                   out.output_handle)) {
    return false;
  }
  return BindingReady(out.input_handle) && BindingReady(out.output_handle) &&
         (!bounded || BindingReady(out.logical_count_handle));
}

} // namespace rund::node::accel::detail
