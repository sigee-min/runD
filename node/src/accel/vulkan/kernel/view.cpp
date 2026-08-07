#include "view.hpp"

#include "../../kernel/backend/exception.hpp"
#include "../../kernel/preparation.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../buffer/resident/find.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"
#include "pipeline/source_artifact.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

inline constexpr std::uint32_t kViewDescriptorCount = 2u;
inline constexpr std::uint32_t kViewBlockSize = 256u;

struct VulkanViewParams final {
  std::uint64_t count{};
  std::uint64_t source_offset_words{};
  std::uint64_t source_stride_words{};
  std::uint64_t target_offset_words{};
  std::uint64_t target_stride_words{};
  std::uint32_t element_words{};
  std::uint32_t reserved{};
};

static_assert(sizeof(VulkanViewParams) == 48u);

[[nodiscard]] constexpr std::string_view VulkanViewSource() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(push_constant) uniform Params {
  uint64_t count;
  uint64_t source_offset_words;
  uint64_t source_stride_words;
  uint64_t target_offset_words;
  uint64_t target_stride_words;
  uint element_words;
  uint reserved;
} params;
layout(set = 0, binding = 0, std430) readonly buffer Source {
  uint source_words[];
};
layout(set = 0, binding = 1, std430) buffer Target {
  uint target_words[];
};
void main() {
  const uint64_t group =
      uint64_t(gl_WorkGroupID.x) +
      uint64_t(gl_WorkGroupID.y) * uint64_t(gl_NumWorkGroups.x);
  const uint64_t index =
      group * uint64_t(gl_WorkGroupSize.x) +
      uint64_t(gl_LocalInvocationID.x);
  if (index >= params.count) { return; }
  const uint64_t source = params.source_offset_words +
                          index * params.source_stride_words;
  const uint64_t target = params.target_offset_words +
                          index * params.target_stride_words;
  target_words[uint(target)] = source_words[uint(source)];
  if (params.element_words == 2u) {
    target_words[uint(target + 1ul)] = source_words[uint(source + 1ul)];
  }
}
)GLSL";
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanViewPipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan pseudo{
      .op_hash_hi = 0x7069706576696577ull,
      .op_hash_lo = 0x636f707975333200ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact =
      VulkanFixedSourceArtifact(VulkanViewSource());
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kViewDescriptorCount,
                                         sizeof(VulkanViewParams), pseudo,
                                         artifact);
}

namespace {

[[nodiscard]] bool Strided(const rund::kernel::ResidentBufferRef &ref) {
  // Stride cannot change the selected address set for zero or one element.
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

[[nodiscard]] VulkanResidentBufferResult
ResolveDense(const rund::AccelDevice &pick, const std::uint64_t binding,
             const rund::kernel::ResidentBufferRef &requested,
             const KernelPreparationMode mode,
             const KernelViewLayout *const views,
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

[[nodiscard]] rund::AccelCheck EncodeTransfers(const VulkanViewLowering &view,
                                               const VkCommandBuffer command,
                                               const bool inputs) {
  if (view.transfers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (command == VK_NULL_HANDLE || view.pipeline == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  bool encoded = false;
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     view.pipeline->pipeline);
  for (const VulkanViewTransfer &transfer : view.transfers) {
    if (transfer.input != inputs) {
      continue;
    }
    const std::uint64_t element_words =
        transfer.element_bytes / sizeof(std::uint32_t);
    const std::uint64_t stride_words =
        transfer.stride_bytes / sizeof(std::uint32_t);
    for (const ViewPage &page : transfer.pages) {
      if (page.descriptor == VK_NULL_HANDLE || !page.grid.valid()) {
        return rund::AccelCheck{false, "compute_resident_stride_invalid"};
      }
      BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            view.pipeline->pipeline_layout, 0u, 1u,
                            &page.descriptor, 0u, nullptr);
      const VulkanViewParams params{
          .count = page.count,
          .source_offset_words =
              transfer.input ? page.external_words : page.dense_words,
          .source_stride_words = transfer.input ? stride_words : element_words,
          .target_offset_words =
              transfer.input ? page.dense_words : page.external_words,
          .target_stride_words = transfer.input ? element_words : stride_words,
          .element_words = static_cast<std::uint32_t>(element_words),
      };
      vkCmdPushConstants(command, view.pipeline->pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                         &params);
      DispatchVulkan(command, page.grid.x, page.grid.y, 1u);
      encoded = true;
    }
  }
  const bool closes_input_lifetime = !inputs && view.has_input;
  if (encoded || closes_input_lifetime) {
    EncodeVulkanComputeToComputeBarrier(command);
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

std::string_view VulkanViewSourceText() noexcept { return VulkanViewSource(); }

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
        const Grid grid = PlanGrid(range.count, kViewBlockSize,
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
  if (!view->binds.valid() ||
      !RebindBoundStep(source, view->binds, view->step)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  out = std::move(view);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
PrepareVulkanViewCommands(VulkanAdapter &adapter,
                          const std::shared_ptr<VulkanViewLowering> &view) {
  if (view == nullptr || view->transfers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  const bool borrowed_pipeline = view->pipeline != nullptr;
  if (!borrowed_pipeline) {
    view->pipeline = AcquireVulkanViewPipeline(adapter);
  }
  if (view->pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  const std::uint64_t descriptor_set_count = VulkanViewDispatchCount(view);
  if (!borrowed_pipeline && !ReserveVulkanCollectiveDescriptorDemand(
                                adapter, *view->pipeline, kViewDescriptorCount,
                                descriptor_set_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  for (VulkanViewTransfer &transfer : view->transfers) {
    const VulkanBuffer *const external = transfer.external.device_buffer;
    const VulkanBuffer *const dense = transfer.dense.device_buffer;
    if (external == nullptr || dense == nullptr) {
      return rund::AccelCheck{false, "accel_buffer_unavailable"};
    }
    const VulkanStorageBinding dense_binding =
        VulkanStorageBindingFor(dense, transfer.dense.ref);
    for (ViewPage &page : transfer.pages) {
      if (!AcquireVulkanCollectiveDescriptorSet(adapter, *view->pipeline,
                                                kViewDescriptorCount,
                                                page.descriptor)) {
        return rund::AccelCheck{false, VulkanLastError(&adapter)};
      }
      const VulkanStorageBinding external_binding{external, page.base_bytes,
                                                  page.span_bytes};
      const std::array<VulkanStorageBinding, kViewDescriptorCount> bindings{
          transfer.input ? external_binding : dense_binding,
          transfer.input ? dense_binding : external_binding};
      if (!WriteVulkanStorageDescriptorSet(adapter, page.descriptor,
                                           bindings)) {
        return rund::AccelCheck{false, VulkanLastError(&adapter)};
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
EncodeVulkanViewInputs(const std::shared_ptr<VulkanViewLowering> &view,
                       const VkCommandBuffer command) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, command, true);
}

rund::AccelCheck
EncodeVulkanViewOutputs(const std::shared_ptr<VulkanViewLowering> &view,
                        const VkCommandBuffer command) {
  return view == nullptr ? rund::AccelCheck{true, "ok"}
                         : EncodeTransfers(*view, command, false);
}

PreparedMemory VulkanViewMemory(const std::shared_ptr<VulkanViewLowering> &view,
                                const std::uint64_t budget,
                                std::uint64_t &traffic) noexcept {
  std::uint64_t bytes = 0u;
  traffic = 0u;
  if (view != nullptr) {
    for (const VulkanViewTransfer &transfer : view->transfers) {
      const std::uint64_t dense = transfer.count * transfer.element_bytes;
      traffic = ::rund::detail::counter::SaturatingAdd(traffic, dense);
      if (transfer.planned) {
        continue;
      }
      bytes = bytes > std::numeric_limits<std::uint64_t>::max() - dense
                  ? std::numeric_limits<std::uint64_t>::max()
                  : bytes + dense;
    }
  }
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = budget};
}

std::uint64_t VulkanViewDispatchCount(
    const std::shared_ptr<VulkanViewLowering> &view) noexcept {
  std::uint64_t count = 0u;
  if (view != nullptr) {
    for (const VulkanViewTransfer &transfer : view->transfers) {
      count = ::rund::detail::counter::SaturatingAdd(
          count, static_cast<std::uint64_t>(transfer.pages.size()));
    }
  }
  return count;
}

#endif

} // namespace rund::node::accel::detail
