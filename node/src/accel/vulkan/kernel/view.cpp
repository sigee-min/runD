#include "view.hpp"

#include "../../kernel/backend/exception.hpp"
#include "../../kernel/preparation.hpp"
#include "../adapter/access.hpp"
#include "../buffer/resident/create.hpp"
#include "../buffer/resident/find.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"
#include "view/internal.hpp"

#include <limits>
#include <mutex>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool Strided(const rund::kernel::ResidentBufferRef &ref) {
  return ref.count > 1u && ref.stride_bytes != ref.element_bytes;
}

struct Replacement final {
  VulkanResidentBufferResult resident{};
};

[[nodiscard]] VulkanResidentBufferResult
ResolveExternal(const rund::AccelDevice &pick,
                const rund::kernel::ResidentBufferRef &ref,
                const std::shared_ptr<void> &handle) {
  if (!VulkanPickOwnsAdapter(pick)) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_buffer_backend_unavailable");
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  VulkanResidentBufferResult result = ResolveVulkanResidentBuffer(
      resident, ref, handle, "compute_resident_id_invalid", true);
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

[[nodiscard]] VulkanResidentBufferResult ResolveDense(
    const rund::AccelDevice &pick, const std::uint64_t binding,
    const rund::kernel::ResidentBufferRef &requested,
    const KernelPreparationMode mode, const KernelViewLayout *const views,
    const RunBinds *const view_binds, bool &planned) {
  planned = views != nullptr || view_binds != nullptr;
  if (!planned) {
    if (IsPipelinePrivatePreparation(mode)) {
      return RejectResident<VulkanResidentBufferResult>(
          "compute_pipeline_memory_plan_invalid");
    }
    const std::uint64_t bytes = requested.count * requested.element_bytes;
    return CreateVulkanResidentBuffer(
        pick,
        ResidentDesc{.bytes = bytes,
                     .element_bytes = requested.element_bytes,
                     .stride_bytes = requested.element_bytes,
                     .count = requested.count,
                     .usage = requested.usage},
        false);
  }
  if (views == nullptr || view_binds == nullptr || !view_binds->valid()) {
    return RejectResident<VulkanResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  const std::uint64_t bytes = requested.count * requested.element_bytes;
  const KernelViewSlot *const slot =
      FindKernelViewSlot(*views, binding, requested);
  if (slot == nullptr || slot->slot >= view_binds->size()) {
    return RejectResident<VulkanResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  rund::kernel::ResidentBufferRef ref = view_binds->refs()[slot->slot];
  if (ref.offset_bytes > ref.bytes || bytes > ref.bytes - ref.offset_bytes) {
    return RejectResident<VulkanResidentBufferResult>(
        "compute_pipeline_memory_plan_invalid");
  }
  ref.element_bytes = requested.element_bytes;
  ref.stride_bytes = requested.element_bytes;
  ref.count = requested.count;
  ref.usage = requested.usage;
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_buffer_backend_unavailable");
  }
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  VulkanResidentBufferResult result = ResolveVulkanResidentBuffer(
      resident, ref, view_binds->handles()[slot->slot],
      "compute_resident_id_invalid");
  if (result.check.ok) {
    result.ref = ref;
  }
  return result;
}

[[nodiscard]] bool ViewAddressable(const rund::kernel::ResidentBufferRef &ref,
                                   const StorageRange range) noexcept {
  if ((ref.element_bytes != 4u && ref.element_bytes != 8u) ||
      ref.offset_bytes % sizeof(std::uint32_t) != 0u ||
      ref.stride_bytes % sizeof(std::uint32_t) != 0u || range.count == 0u ||
      range.offset % sizeof(std::uint32_t) != 0u) {
    return false;
  }
  const std::uint64_t words = ref.element_bytes / sizeof(std::uint32_t);
  const std::uint64_t offset = range.offset / sizeof(std::uint32_t);
  const std::uint64_t stride = ref.stride_bytes / sizeof(std::uint32_t);
  constexpr std::uint64_t limit = std::numeric_limits<std::uint32_t>::max();
  if (range.count > limit || offset > limit || words - 1u > limit - offset) {
    return false;
  }
  return range.count - 1u <= (limit - offset - (words - 1u)) / stride;
}

[[nodiscard]] bool MapReady(const BoundStep &step,
                            const rund::kernel::ResidentBufferRef &ref,
                            const std::uint64_t alignment) noexcept {
  if (step.step == nullptr ||
      step.step->kind() != rund::kernel::NodeKind::Map || alignment == 0u ||
      step.map_windows.size() == 0u || ref.stride_bytes == 0u) {
    return false;
  }
  const std::uint64_t bias = ref.offset_bytes % alignment;
  for (std::uint64_t index = 0u; index < step.map_windows.size(); ++index) {
    const rund::kernel::ComputeDispatchWindow window =
        step.map_windows.data()[index];
    if (window.begin_sequence >
            std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
        ref.offset_bytes > std::numeric_limits<std::uint64_t>::max() -
                               window.begin_sequence * ref.stride_bytes ||
        (ref.offset_bytes + window.begin_sequence * ref.stride_bytes) %
                alignment !=
            bias) {
      return false;
    }
  }
  return true;
}

} // namespace

rund::AccelCheck PrepareVulkanViewLowering(
    const rund::AccelDevice &pick, const BoundStep &source,
    const KernelPreparationMode mode, const KernelViewLayout *const views,
    const RunBinds *const view_binds,
    std::shared_ptr<VulkanViewLowering> &out) {
  out.reset();
  if (source.step == nullptr || source.source_binds == nullptr ||
      source.step->kind() == rund::kernel::NodeKind::Map ||
      source.step->kind() == rund::kernel::NodeKind::ScatterReduce) {
    return rund::AccelCheck{true, "ok"};
  }
  const RunBinds &original = *source.source_binds;
  if (!original.valid() || !VulkanPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<VulkanViewLowering> view;
  try {
    view = std::make_shared<VulkanViewLowering>();
    view->transfers.reserve(source.step->graph_binding_indices.size());
    view->transfer_by_binding.resize(original.size(), 0u);
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  view->adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::vector<Replacement> replacements;
  std::vector<std::uint32_t> replacement_by_binding;
  try {
    replacements.reserve(source.step->graph_binding_indices.size());
    replacement_by_binding.resize(original.size(), 0u);
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t local = 0u;
       local < source.step->graph_binding_indices.size(); ++local) {
    const std::uint64_t index = source.step->graph_binding_indices[local];
    if (index >= original.size() ||
        replacement_by_binding[static_cast<std::size_t>(index)] != 0u) {
      continue;
    }
    const rund::kernel::ResidentBufferRef &ref = original.refs()[index];
    if (MapReady(source, ref, view->adapter->storage_align)) {
      continue;
    }
    const bool normalize_singleton =
        ref.count == 1u && ref.stride_bytes != ref.element_bytes;
    const bool descriptor_ready =
        ref.offset_bytes % view->adapter->storage_align == 0u;
    if (!Strided(ref) && !normalize_singleton && descriptor_ready) {
      continue;
    }
    if (ref.count == 0u || ref.element_bytes == 0u ||
        ref.count >
            std::numeric_limits<std::uint64_t>::max() / ref.element_bytes ||
        ref.count * ref.element_bytes > view->adapter->storage_limit) {
      return rund::AccelCheck{false, "compute_resident_stride_invalid"};
    }
    VulkanResidentBufferResult external =
        ResolveExternal(pick, ref, original.handles()[index]);
    if (!external.check.ok) {
      return external.check;
    }
    if (normalize_singleton && descriptor_ready) {
      external.ref.stride_bytes = external.ref.element_bytes;
      try {
        replacements.push_back(Replacement{.resident = std::move(external)});
        replacement_by_binding[static_cast<std::size_t>(index)] =
            static_cast<std::uint32_t>(replacements.size());
      } catch (...) {
        backend_exception::RethrowUnlessCapacityException();
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }
    bool planned = false;
    VulkanResidentBufferResult dense =
        ResolveDense(pick, index, ref, mode, views, view_binds, planned);
    if (!dense.check.ok) {
      return dense.check;
    }
    VulkanViewTransfer transfer{
        .binding = index,
        .external = std::move(external),
        .dense = dense,
        .count = ref.count,
        .element_bytes = ref.element_bytes,
        .offset_bytes = ref.offset_bytes,
        .stride_bytes = ref.stride_bytes,
        .input = ref.usage == rund::kernel::kResidentUsageRead,
        .planned = planned,
    };
    std::uint64_t begin = 0u;
    try {
      while (begin < ref.count) {
        StorageRange range{};
        if (!PlanStoragePage(*view->adapter, ref, begin, range) ||
            !ViewAddressable(ref, range)) {
          return rund::AccelCheck{false, "compute_resident_stride_invalid"};
        }
        const Grid grid =
            PlanGrid(range.count, kVulkanViewBlockSize,
                     view->adapter->max_dispatch_groups,
                     view->adapter->dispatch_rows);
        if (!grid.valid()) {
          return rund::AccelCheck{false, "compute_resident_stride_invalid"};
        }
        transfer.pages.push_back(ViewPage{
            .begin = begin,
            .count = range.count,
            .base_bytes = range.base,
            .span_bytes = range.bytes,
            .external_words = range.offset / sizeof(std::uint32_t),
            .dense_words = begin * (ref.element_bytes / sizeof(std::uint32_t)),
            .grid = grid,
        });
        begin += range.count;
      }
      view->transfers.push_back(std::move(transfer));
      view->has_input = view->has_input || view->transfers.back().input;
      replacements.push_back(Replacement{.resident = std::move(dense)});
      view->transfer_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(view->transfers.size());
      replacement_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(replacements.size());
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (replacements.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  view->binds.reserve(original.size());
  for (std::uint64_t index = 0u; index < original.size(); ++index) {
    const std::uint32_t ordinal =
        replacement_by_binding[static_cast<std::size_t>(index)];
    const Replacement *const replacement =
        ordinal == 0u ? nullptr : &replacements[ordinal - 1u];
    if (!view->binds.push(replacement == nullptr ? original.refs()[index]
                                                 : replacement->resident.ref,
                          replacement == nullptr
                              ? original.handles()[index]
                              : replacement->resident.handle)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!view->binds.valid() || !RebindBoundStep(source, view->binds, view->step)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  out = std::move(view);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
