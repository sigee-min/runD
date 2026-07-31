#include "local.hpp"

#include "../../context/internal.hpp"
namespace rund::node::accel::detail {

bool BuildSortBinds(const KernelExecutionStep &step, const RunBinds &run_binds,
                    SortBinds &out) {
  const auto &active = step.operation.get<operation::Sort>();
  const bool identity_values =
      active.plan.value == rund::kernel::SortValue::IdentityU32;
  const bool bounded =
      active.plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const std::uint64_t expected_bindings =
      (identity_values ? 3u : 4u) + (bounded ? 1u : 0u);
  if (!step.graph_binding_indices_ok ||
      step.graph_binding_indices.size() != expected_bindings) {
    return false;
  }

  const BindingSource source = BindingSourceFor(run_binds);
  std::size_t index = 0u;
  if (!ReadBinding(source, step.graph_binding_indices[index++], out.read_keys,
                   out.read_keys_handle)) {
    return false;
  }
  if (bounded && !ReadBinding(source, step.graph_binding_indices[index++],
                              out.logical_count, out.logical_count_handle)) {
    return false;
  }
  if (!identity_values &&
      !ReadBinding(source, step.graph_binding_indices[index++], out.read_values,
                   out.read_values_handle)) {
    return false;
  }
  if (!ReadBinding(source, step.graph_binding_indices[index++], out.write_keys,
                   out.write_keys_handle) ||
      !ReadBinding(source, step.graph_binding_indices[index], out.write_values,
                   out.write_values_handle)) {
    return false;
  }

  return BindingReady(out.read_keys_handle) &&
         (identity_values || BindingReady(out.read_values_handle)) &&
         (!bounded || BindingReady(out.logical_count_handle)) &&
         BindingReady(out.write_keys_handle) &&
         BindingReady(out.write_values_handle);
}

} // namespace rund::node::accel::detail
