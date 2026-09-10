#include "local.hpp"

#include "../../../../map/local.hpp"

namespace rund::node::accel::detail::graph_direct_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool GraphDirectReduceBindings(const KernelExecution &execution,
                                             const BoundStep &step,
                                             const RunBinds &binds) noexcept {
  const operation::Reduce *const operation =
      OperationFor<operation::Reduce>(step);
  const ReduceBinds *const bindings =
      BindingsFor<ReduceBinds>(step, rund::kernel::NodeKind::Reduce);
  if (operation == nullptr || bindings == nullptr ||
      operation->desc.element != rund::kernel::ReduceElement::U64 ||
      operation->desc.op != rund::kernel::ReduceOp::Sum ||
      !GraphDirectResidentRef(bindings->input, bindings->input_handle,
                              rund::kernel::kResidentUsageRead) ||
      !GraphDirectResidentRef(bindings->output, bindings->output_handle,
                              rund::kernel::kResidentUsageWrite) ||
      (bindings->logical_count != nullptr &&
       !GraphDirectResidentRef(bindings->logical_count,
                               bindings->logical_count_handle,
                               rund::kernel::kResidentUsageRead))) {
    return false;
  }
  const auto &accesses = step.step->artifact.metadata.binding_accesses;
  if (accesses.size() != step.step->graph_binding_indices.size()) {
    return false;
  }
  std::size_t read_position = 0u;
  bool wrote_output = false;
  for (std::size_t position = 0u; position < accesses.size(); ++position) {
    const std::size_t binding = step.step->graph_binding_indices[position];
    if (accesses[position] == rund::kernel::ComputeBindingAccess::Read) {
      const rund::kernel::ResidentBufferRef *ref = nullptr;
      const std::shared_ptr<void> *handle = nullptr;
      if (read_position == 0u) {
        ref = bindings->input;
        handle = bindings->input_handle;
      } else if (read_position == 1u && bindings->logical_count != nullptr) {
        ref = bindings->logical_count;
        handle = bindings->logical_count_handle;
      } else {
        return false;
      }
      if (execution.graph_roles[binding] != rund::kernel::BufferRole::Read ||
          !GraphDirectCanonicalResidentRef(binds, ref, handle, binding,
                                           rund::kernel::kResidentUsageRead)) {
        return false;
      }
      ++read_position;
    } else if (accesses[position] ==
               rund::kernel::ComputeBindingAccess::Write) {
      if (wrote_output ||
          execution.graph_roles[binding] != rund::kernel::BufferRole::Write ||
          !GraphDirectCanonicalResidentRef(binds, bindings->output,
                                           bindings->output_handle, binding,
                                           rund::kernel::kResidentUsageWrite)) {
        return false;
      }
      wrote_output = true;
    } else {
      return false;
    }
  }
  return wrote_output &&
         read_position == (bindings->logical_count == nullptr ? 1u : 2u);
}

#endif

} // namespace rund::node::accel::detail::graph_direct_detail
