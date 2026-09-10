#include "local.hpp"

#include "../../../../map/local.hpp"

namespace rund::node::accel::detail::graph_direct_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool GraphDirectMapBindings(const KernelExecution &execution,
                                          const BoundStep &step,
                                          const RunBinds &binds) noexcept {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || !bindings->valid() ||
      bindings->inputs.size() == 0u || bindings->outputs.size() != 1u) {
    return false;
  }
  const rund::kernel::ResidentBindingRange inputs = bindings->inputs.range();
  const rund::kernel::ResidentBindingRange outputs = bindings->outputs.range();
  const rund::kernel::ResidentBufferRef *const refs = binds.refs();
  const std::shared_ptr<void> *const handles = binds.handles();
  if (step.step->artifact.metadata.read_count == 0u ||
      step.step->artifact.metadata.write_count != 1u || inputs.refs != refs ||
      inputs.handles != handles || inputs.storage_count != binds.size() ||
      outputs.refs != refs || outputs.handles != handles ||
      outputs.storage_count != binds.size() ||
      step.step->artifact.metadata.binding_accesses.size() >
          step.step->graph_binding_indices.size()) {
    return false;
  }
  const auto &accesses = step.step->artifact.metadata.binding_accesses;
  std::size_t read_position = 0u;
  std::size_t write_position = 0u;
  for (std::size_t position = 0u; position < accesses.size(); ++position) {
    const std::size_t binding = step.step->graph_binding_indices[position];
    if (accesses[position] == rund::kernel::ComputeBindingAccess::Read) {
      if (read_position >= inputs.count ||
          execution.graph_roles[binding] != rund::kernel::BufferRole::Read ||
          !GraphDirectCanonicalResidentRef(
              binds, inputs.ref(read_position), inputs.handle(read_position),
              binding, rund::kernel::kResidentUsageRead)) {
        return false;
      }
      ++read_position;
    } else if (accesses[position] ==
               rund::kernel::ComputeBindingAccess::Write) {
      if (write_position >= outputs.count ||
          execution.graph_roles[binding] != rund::kernel::BufferRole::Write ||
          !GraphDirectCanonicalResidentRef(binds, outputs.ref(write_position),
                                           outputs.handle(write_position),
                                           binding,
                                           rund::kernel::kResidentUsageWrite)) {
        return false;
      }
      ++write_position;
    } else {
      return false;
    }
  }
  return read_position == inputs.count && write_position == outputs.count &&
         read_position == step.step->artifact.metadata.read_count &&
         write_position == 1u;
}

#endif

} // namespace rund::node::accel::detail::graph_direct_detail
