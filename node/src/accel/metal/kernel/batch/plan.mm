#include "plan.hpp"

#include "../../../kernel/backend/execute.hpp"
#include "../../number.hpp"
#include "../../runtime/map/api.hpp"
#include "../local.hpp"

#include <limits>
#include <new>
#include <utility>

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

constexpr std::uint64_t Alignment = 256u;

[[nodiscard]] constexpr bool same(const EntryKey &left,
                                  const EntryKey &right) noexcept {
  return left.steps == right.steps && left.artifact == right.artifact &&
         left.kernel == right.kernel && left.tiles == right.tiles &&
         left.step_count == right.step_count &&
         left.input_count == right.input_count &&
         left.output_count == right.output_count &&
         left.requires_reset == right.requires_reset;
}

[[nodiscard]] EntryKey key(const BackendBatchEntry &entry) noexcept {
  EntryKey result{};
  if (entry.run == nullptr || entry.run->execution == nullptr) {
    return result;
  }
  result.steps = entry.run->execution->steps.data();
  result.kernel = entry.run->execution->admission.kernel_id;
  result.step_count = entry.run->step_count;
  result.requires_reset =
      entry.run->resets != nullptr && !entry.run->resets->empty();
  if (entry.run->step_count == 1u && entry.run->steps != nullptr &&
      entry.run->steps[0].planned != nullptr) {
    const PlannedStep &planned = *entry.run->steps[0].planned;
    result.artifact = planned.artifact;
    result.tiles = planned.plan.tile_count;
    result.input_count = planned.plan.input_buffer_count;
    result.output_count = planned.plan.output_buffer_count;
  }
  return result;
}

[[nodiscard]] MetalMapEncodeResources *
resolve(const BackendBatchEntry &entry) noexcept {
  auto *const resources =
      entry.prepared == nullptr
          ? nullptr
          : static_cast<MetalKernelResources *>(entry.prepared->get());
  if (entry.run == nullptr || entry.run->execution == nullptr ||
      entry.run->step_count != 1u || entry.run->steps == nullptr ||
      entry.run->steps[0].step == nullptr ||
      entry.run->steps[0].step->kind() != rund::kernel::NodeKind::Map ||
      resources == nullptr || resources->size() != 1u) {
    return nullptr;
  }
  MetalKernelEntry *const step = resources->entry(0u);
  return step == nullptr || step->ops.encode != EncodeMetalMap
             ? nullptr
             : static_cast<MetalMapEncodeResources *>(step->resource.get());
}

[[nodiscard]] BatchMapView view(const BackendBatchEntry &entry,
                                MetalMapEncodeResources *const map) noexcept {
  BatchMapView result{};
  const BoundStep *const bound =
      entry.run == nullptr || entry.run->steps == nullptr
          ? nullptr
          : &entry.run->steps[0];
  if (map == nullptr || bound == nullptr || bound->planned == nullptr ||
      bound->planned->artifact == nullptr ||
      bound->planned->artifact->metadata.uses_index ||
      map->windows.size() != 1u || map->windows.front().begin_sequence != 0u ||
      map->windows.front().tile_count != map->plan.tile_count ||
      map->plan.tile_count == 0u ||
      map->plan.input_buffer_count > result.inputs.size() ||
      map->plan.output_buffer_count == 0u ||
      map->plan.output_buffer_count > result.outputs.size() ||
      entry.run->execution->admission.kernel_id == 0u ||
      (entry.run->resets != nullptr && !entry.run->resets->empty())) {
    return result;
  }
  result.kernel = entry.run->execution->admission.kernel_id;
  result.tiles = map->plan.tile_count;
  result.input_count = static_cast<std::size_t>(map->plan.input_buffer_count);
  result.output_count = static_cast<std::size_t>(map->plan.output_buffer_count);
  result.max_tiles = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t index = 0u; index < result.input_count; ++index) {
    const auto *const ref = map->bindings.resident_inputs.ref(index);
    const MetalResidentBufferResult &buffer = map->resident.input(index);
    if (ref == nullptr || ref->id == 0u || ref->element_bytes == 0u ||
        ref->stride_bytes != ref->element_bytes || ref->count < result.tiles ||
        !rund::kernel::checked::mul(result.tiles, ref->element_bytes) ||
        result.tiles * ref->element_bytes > ref->bytes || !buffer.check.ok ||
        buffer.device_buffer == nullptr ||
        MetalBufferContents(MetalRuntimeBuffer{
            .bytes = ref->bytes, .buffer = buffer.device_buffer}) == nullptr) {
      return BatchMapView{};
    }
    result.inputs[index] = ref->element_bytes;
    result.input_ids[index] = ref->id;
  }
  for (std::size_t index = 0u; index < result.output_count; ++index) {
    const auto *const ref = map->bindings.resident_outputs.ref(index);
    const MetalResidentBufferResult &buffer = map->resident.output(index);
    if (ref == nullptr || ref->id == 0u || ref->element_bytes == 0u ||
        ref->stride_bytes != ref->element_bytes || ref->count < result.tiles ||
        !rund::kernel::checked::mul(result.tiles, ref->element_bytes) ||
        result.tiles * ref->element_bytes > ref->bytes || !buffer.check.ok ||
        buffer.device_buffer == nullptr ||
        MetalBufferContents(MetalRuntimeBuffer{
            .bytes = ref->bytes, .buffer = buffer.device_buffer}) == nullptr) {
      return BatchMapView{};
    }
    result.outputs[index] = ref->element_bytes;
    result.output_ids[index] = ref->id;
  }
  result.eligible = true;
  return result;
}

[[nodiscard]] bool matches(const Workspace &workspace,
                           const std::array<EntryKey, BatchMapCapacity> &keys,
                           const std::size_t size) noexcept {
  if (!workspace.planned || workspace.size != size) {
    return false;
  }
  for (std::size_t index = 0u; index < size; ++index) {
    if (!same(workspace.entries[index], keys[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool open(MetalAdapter &adapter, std::shared_ptr<void> &owner,
                        Workspace *&out) {
  if (owner == nullptr) {
    try {
      auto made = std::make_shared<Workspace>();
      made->adapter = &adapter;
      owner = std::static_pointer_cast<void>(made);
    } catch (const std::bad_alloc &) {
      return false;
    }
  }
  out = static_cast<Workspace *>(owner.get());
  return out != nullptr && out->adapter == &adapter;
}

[[nodiscard]] bool ensure(MetalAdapter &adapter, const BatchMapPlan &plan,
                          Workspace &workspace) {
  const auto buffer = [&](MetalRuntimeBuffer &current,
                          const std::uint64_t bytes,
                          const MetalBufferUsage usage) {
    if (bytes == 0u) {
      return true;
    }
    if (current.buffer != nullptr && current.bytes >= bytes &&
        MetalBufferContents(current) != nullptr) {
      return true;
    }
    if (current.buffer != nullptr) {
      ReleaseMetalBuffer(adapter, std::move(current));
      current = {};
    }
    current = AcquireMetalBuffer(adapter, bytes, usage);
    return current.buffer != nullptr && current.bytes >= bytes &&
           MetalBufferContents(current) != nullptr;
  };
  return buffer(workspace.input, plan.input_bytes, MetalBufferUsage::Input) &&
         buffer(workspace.output, plan.output_bytes, MetalBufferUsage::Output);
}

} // namespace

Workspace::~Workspace() {
  if (adapter == nullptr) {
    return;
  }
  ReleaseMetalBuffer(*adapter, std::move(input));
  ReleaseMetalBuffer(*adapter, std::move(output));
}

rund::AccelCheck prepare(MetalAdapter &adapter,
                         const std::span<const BackendBatchEntry> entries,
                         std::shared_ptr<void> &owner, Workspace *&workspace,
                         Maps &maps) {
  if (entries.size() > BatchMapCapacity || !open(adapter, owner, workspace)) {
    return rund::AccelCheck{false, "compute_batch_workspace_invalid"};
  }
  std::array<EntryKey, BatchMapCapacity> keys{};
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    keys[index] = key(entries[index]);
    maps[index] = resolve(entries[index]);
  }
  if (!matches(*workspace, keys, entries.size())) {
    std::array<BatchMapView, BatchMapCapacity> views{};
    for (std::size_t index = 0u; index < entries.size(); ++index) {
      views[index] = view(entries[index], maps[index]);
    }
    const BatchMapPlan plan = PlanBatchMaps(
        std::span<const BatchMapView>{views.data(), entries.size()}, Alignment,
        adapter.caps.staging_bytes);
    if (!plan.ok) {
      return rund::AccelCheck{false, plan.reason};
    }
    workspace->views = views;
    workspace->entries = keys;
    workspace->plan = plan;
    workspace->size = entries.size();
    workspace->planned = true;
  }
  return ensure(adapter, workspace->plan, *workspace)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_batch_workspace_invalid"};
}

#endif

} // namespace rund::node::accel::detail::metalbatch
