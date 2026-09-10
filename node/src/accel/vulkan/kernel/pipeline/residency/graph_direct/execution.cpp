#include "local.hpp"

namespace rund::node::accel::detail::graph_direct_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool GraphDirectExecution(const BackendRun &run,
                                        const BoundStep &bound,
                                        const RunBinds &binds) noexcept {
  const KernelExecution *const execution = run.execution;
  if (execution == nullptr || !execution->admission.check.ok ||
      execution->admission.graph_id_hi == 0u ||
      execution->admission.graph_id_lo == 0u ||
      execution->admission.node_count == 0u ||
      execution->admission.scalar != rund::kernel::ComputeScalar::Lane64 ||
      execution->admission.domain != rund::kernel::ComputeDomain::U64 ||
      !binds.valid() || execution->graph_roles.size() != binds.size() ||
      execution->graph_shapes.size() != binds.size() ||
      execution->graph_visibilities.size() != binds.size() ||
      execution->graph_alias_representatives.size() != binds.size() ||
      execution->steps.size() != run.step_count || run.steps == nullptr) {
    return false;
  }
  const rund::kernel::ResidentBufferRef *const refs = binds.refs();
  const std::shared_ptr<void> *const handles = binds.handles();
  if (refs == nullptr || handles == nullptr) {
    return false;
  }
  for (std::size_t index = 0u; index < binds.size(); ++index) {
    const rund::kernel::BufferRole role = execution->graph_roles[index];
    const std::uint32_t usage = role == rund::kernel::BufferRole::Read
                                    ? rund::kernel::kResidentUsageRead
                                : role == rund::kernel::BufferRole::Write
                                    ? rund::kernel::kResidentUsageWrite
                                    : 0u;
    const rund::AccelBufferDesc &shape = execution->graph_shapes[index];
    if (usage == 0u || shape.scalar_width_bytes != sizeof(std::uint64_t) ||
        shape.count == 0u || refs[index].count != shape.count ||
        !GraphDirectResidentRef(&refs[index], &handles[index], usage)) {
      return false;
    }
    const std::uint64_t representative =
        execution->graph_alias_representatives[index];
    if (representative >= binds.size() || representative > index ||
        (representative != index &&
         (refs[representative].id != refs[index].id ||
          refs[representative].offset_bytes != refs[index].offset_bytes ||
          refs[representative].element_bytes != refs[index].element_bytes ||
          refs[representative].stride_bytes != refs[index].stride_bytes ||
          refs[representative].count != refs[index].count ||
          refs[representative].bytes != refs[index].bytes ||
          handles[representative].get() != handles[index].get()))) {
      return false;
    }
  }
  const BoundStep &step = bound;
  if (step.step == nullptr || step.planned == nullptr ||
      step.source_binds != &binds || !step.step->graph_binding_indices_ok ||
      !step.step->graph_binding_indices.valid() ||
      step.step->artifact.key.scalar != rund::kernel::ComputeScalar::Lane64 ||
      step.step->artifact.key.domain != rund::kernel::ComputeDomain::U64) {
    return false;
  }
  const auto &accesses = step.step->artifact.metadata.binding_accesses;
  if (accesses.empty() ||
      accesses.size() > step.step->graph_binding_indices.size()) {
    return false;
  }
  for (std::size_t binding = 0u; binding < accesses.size(); ++binding) {
    if (step.step->graph_binding_indices[binding] >= binds.size()) {
      return false;
    }
  }
  const auto control_binding =
      [&](const std::uint32_t local,
          const rund::kernel::ResidentBufferRef *const ref,
          const std::shared_ptr<void> *const handle) {
        if (local >= step.step->graph_binding_indices.size()) {
          return false;
        }
        const std::uint64_t binding = step.step->graph_binding_indices[local];
        return binding < binds.size() &&
               execution->graph_roles[binding] ==
                   rund::kernel::BufferRole::Read &&
               GraphDirectCanonicalResidentRef(
                   binds, ref, handle, binding,
                   rund::kernel::kResidentUsageRead);
      };
  if (!step.control.control.valid(binds.size()) ||
      (step.control.count == nullptr) != !step.control.control.has_count() ||
      (step.control.predicate == nullptr) !=
          !step.control.control.has_predicate() ||
      (step.control.count != nullptr &&
       !control_binding(step.control.control.count_binding, step.control.count,
                        step.control.count_handle)) ||
      (step.control.predicate != nullptr &&
       !control_binding(step.control.control.predicate_binding,
                        step.control.predicate,
                        step.control.predicate_handle))) {
    return false;
  }
  if (step.step->kind() == rund::kernel::NodeKind::Map) {
    return GraphDirectMapBindings(*execution, step, binds);
  }
  if (step.step->kind() == rund::kernel::NodeKind::Reduce) {
    return GraphDirectReduceBindings(*execution, step, binds);
  }
  return false;
}

#endif

} // namespace rund::node::accel::detail::graph_direct_detail
