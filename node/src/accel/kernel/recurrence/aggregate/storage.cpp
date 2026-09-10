#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::nested_aggregate_detail {

using rund::kernel::ComputeDomain;
using rund::kernel::ComputeScalar;

bool View::valid() const noexcept {
  return ref != nullptr && handle != nullptr && *handle != nullptr;
}

View At(const ResidentBindingRange &range, const std::uint64_t index) noexcept {
  return View{.ref = range.ref(index), .handle = range.handle(index)};
}

View At(const ResidentBufferRef *const ref,
        const std::shared_ptr<void> *const handle) noexcept {
  return View{.ref = ref, .handle = handle};
}

bool SameStorage(const View left, const View right) noexcept {
  return left.valid() && right.valid() && left.ref->id == right.ref->id &&
         left.ref->bytes == right.ref->bytes &&
         left.ref->offset_bytes == right.ref->offset_bytes &&
         left.ref->element_bytes == right.ref->element_bytes &&
         left.ref->stride_bytes == right.ref->stride_bytes &&
         left.ref->count == right.ref->count && *left.handle == *right.handle;
}

bool ReadView(const View view) noexcept {
  return view.valid() && view.ref->usage == rund::kernel::kResidentUsageRead;
}

bool WriteView(const View view) noexcept {
  return view.valid() && view.ref->usage == rund::kernel::kResidentUsageWrite;
}

bool U32View(const View view, const std::uint64_t count) noexcept {
  return view.valid() && view.ref->element_bytes == sizeof(std::uint32_t) &&
         view.ref->stride_bytes >= sizeof(std::uint32_t) &&
         view.ref->count == count &&
         (view.ref->offset_bytes % sizeof(std::uint32_t)) == 0u &&
         (view.ref->stride_bytes % sizeof(std::uint32_t)) == 0u;
}

bool DenseU32Workspace(const View view, const std::uint64_t count) noexcept {
  if (!WriteView(view) || count == 0u || view.ref->id == 0u ||
      view.ref->element_bytes != sizeof(std::uint32_t) ||
      view.ref->stride_bytes != sizeof(std::uint32_t) ||
      view.ref->count < count ||
      (view.ref->offset_bytes % sizeof(std::uint32_t)) != 0u ||
      view.ref->offset_bytes > view.ref->bytes) {
    return false;
  }
  return view.ref->count <=
         (view.ref->bytes - view.ref->offset_bytes) / sizeof(std::uint32_t);
}

bool InternalOutput(const BackendRun &run, const BoundStep &step) noexcept {
  if (run.execution == nullptr || step.step == nullptr ||
      !step.step->graph_binding_indices_ok ||
      step.step->graph_binding_indices.size() == 0u) {
    return false;
  }
  const std::uint64_t binding =
      step.step
          ->graph_binding_indices[step.step->graph_binding_indices.size() - 1u];
  return binding < run.execution->graph_visibilities.size() &&
         run.execution->graph_visibilities[static_cast<std::size_t>(binding)] ==
             rund::GraphBufferVisibility::Internal;
}

bool DisjointStorage(const View left, const View right) noexcept {
  if (!left.valid() || !right.valid()) {
    return false;
  }
  const bool same_id = left.ref->id == right.ref->id;
  const bool same_owner = *left.handle == *right.handle;
  if (same_id != same_owner) {
    return false;
  }
  if (!same_id) {
    return true;
  }
  const auto end = [](const View view, std::uint64_t &out) {
    std::uint64_t tail = 0u;
    return view.ref->count != 0u &&
           rund::kernel::checked::mul(view.ref->count - 1u,
                                      view.ref->stride_bytes, tail) &&
           rund::kernel::checked::add(view.ref->offset_bytes, tail, out) &&
           rund::kernel::checked::add(out, view.ref->element_bytes, out);
  };
  std::uint64_t left_end = 0u;
  std::uint64_t right_end = 0u;
  return end(left, left_end) && end(right, right_end) &&
         (left_end <= right.ref->offset_bytes ||
          right_end <= left.ref->offset_bytes);
}

NestedAggregateWorkspace Workspace(const View view) {
  return NestedAggregateWorkspace{.ref = *view.ref, .handle = *view.handle};
}

BackendRead Read(const View view) {
  return BackendRead{.source = *view.ref, .handle = *view.handle};
}

bool SameRead(const BackendRead &left, const BackendRead &right) noexcept {
  return SameStorage(At(&left.source, &left.handle),
                     At(&right.source, &right.handle));
}

bool ReadyRun(const BackendRun *const run,
              const std::size_t step_count) noexcept {
  return run != nullptr && run->execution != nullptr && run->steps != nullptr &&
         run->step_count == step_count &&
         (run->resets == nullptr || run->resets->empty());
}

bool ReadyStep(const BoundStep &step, const rund::kernel::NodeKind kind,
               const bool first) noexcept {
  return BoundStepMatches(step, kind) && step.step != nullptr &&
         step.planned != nullptr && step.source_binds != nullptr &&
         !step.control.active() && step.resets.empty() &&
         step.barrier_before == !first;
}

ProgramFingerprint ProgramIdentity(const BackendRun &run) noexcept {
  if (run.execution == nullptr || run.execution->steps.empty()) {
    return {};
  }
  return ProgramFingerprint{
      .api = run.execution->admission.api,
      .graph_id_hi = run.execution->admission.graph_id_hi,
      .graph_id_lo = run.execution->admission.graph_id_lo,
      .scalar = run.execution->admission.scalar,
      .domain = run.execution->admission.domain,
      .fixed_format = run.execution->steps.front().artifact.key.fixed_format,
  };
}

bool SameProgram(const BackendRun &run,
                 const ProgramFingerprint &identity) noexcept {
  if (run.execution == nullptr ||
      run.execution->admission.api != identity.api ||
      run.execution->admission.graph_id_hi != identity.graph_id_hi ||
      run.execution->admission.graph_id_lo != identity.graph_id_lo ||
      run.execution->admission.scalar != identity.scalar ||
      run.execution->admission.domain != identity.domain ||
      run.execution->steps.empty()) {
    return false;
  }
  return run.execution->steps.front().artifact.key.fixed_format ==
         identity.fixed_format;
}

bool U32Program(const ProgramFingerprint &identity) noexcept {
  return identity.api != rund::AccelApi::Cpu &&
         (identity.api == rund::AccelApi::Metal ||
          identity.api == rund::AccelApi::Vulkan) &&
         identity.scalar == ComputeScalar::Lane32 &&
         identity.domain == ComputeDomain::U32 &&
         (identity.graph_id_hi != 0u || identity.graph_id_lo != 0u);
}

} // namespace rund::node::accel::detail::nested_aggregate_detail
