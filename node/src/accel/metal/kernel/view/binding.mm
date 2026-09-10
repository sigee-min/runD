#include "internal.hpp"

#include "../../../kernel/preparation.hpp"
#include "../../buffer/owner.hpp"
#include "../../buffer/resident/find.hpp"
#include "../../resident.hpp"
#include "../../resident/access.hpp"
#include "../../state.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

MetalResidentBufferResult
ResolveMetalViewExternal(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  MetalResidentBufferResult result = ResolveMetalResidentBuffer(
      resident, ref, handle, "accel_metal_resident_id_unavailable", true);
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

MetalResidentBufferResult ResolveMetalViewDense(
    const rund::AccelDevice &pick, const std::uint64_t binding,
    const rund::kernel::ResidentBufferRef &requested,
    const KernelPreparationMode mode, const KernelViewLayout *const views,
    const RunBinds *const view_binds, bool &planned) {
  planned = views != nullptr || view_binds != nullptr;
  if (!planned) {
    if (IsPipelinePrivatePreparation(mode)) {
      return RejectResident<MetalResidentBufferResult>(
          "compute_pipeline_memory_plan_invalid");
    }
    const std::uint64_t bytes = requested.count * requested.element_bytes;
    return CreateMetalResidentBuffer(
        pick,
        ResidentDesc{.bytes = bytes,
                     .element_bytes = requested.element_bytes,
                     .stride_bytes = requested.element_bytes,
                     .count = requested.count,
                     .usage = requested.usage},
        false);
  }
  if (views == nullptr || view_binds == nullptr || !view_binds->valid()) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  const std::uint64_t bytes = requested.count * requested.element_bytes;
  const KernelViewSlot *const slot =
      FindKernelViewSlot(*views, binding, requested);
  if (slot == nullptr || slot->slot >= view_binds->size()) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  rund::kernel::ResidentBufferRef ref = view_binds->refs()[slot->slot];
  if (ref.offset_bytes > ref.bytes || bytes > ref.bytes - ref.offset_bytes) {
    return RejectResident<MetalResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  ref.element_bytes = requested.element_bytes;
  ref.stride_bytes = requested.element_bytes;
  ref.count = requested.count;
  ref.usage = requested.usage;
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return RejectResident<MetalResidentBufferResult>(
        "accel_metal_resident_owner_invalid");
  }
  MetalResidentState &resident = MetalResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  MetalResidentBufferResult result = ResolveMetalResidentBuffer(
      resident, ref, view_binds->handles()[slot->slot],
      "accel_metal_resident_id_unavailable");
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

rund::AccelCheck
BindMetalViewArguments(const RunBinds &original,
                       const std::vector<MetalViewReplacement> &replacements,
                       const std::vector<std::uint32_t> &replacement_by_binding,
                       MetalViewLowering &view) {
  if (replacement_by_binding.size() != original.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  view.binds.reserve(original.size());
  for (std::uint64_t index = 0u; index < original.size(); ++index) {
    const std::uint32_t ordinal =
        replacement_by_binding[static_cast<std::size_t>(index)];
    if (ordinal > replacements.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const MetalViewReplacement *const replacement =
        ordinal == 0u ? nullptr : &replacements[ordinal - 1u];
    if (!view.binds.push(replacement == nullptr ? original.refs()[index]
                                                : replacement->resident.ref,
                         replacement == nullptr
                             ? original.handles()[index]
                             : replacement->resident.handle)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
