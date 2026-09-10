#include "../../../adapter/error.hpp"

#include "internal.hpp"

#include "../../../../kernel/recurrence/source.hpp"
#include "../../../../kernel/step/map/stride.hpp"
#include "../../../buffer/create/telemetry.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanRecurrenceTemplateProbe final {
  const MapRecurrence *recurrence{};
  const BackendRun *signature{};
  const MapRecurrencePreparationPlan *plan{};
  VulkanAdapter *adapter{};
  bool history{};
};

[[nodiscard]] bool
MatchVulkanRecurrenceTemplate(const void *const prepared,
                              const void *const raw_probe) noexcept {
  if (prepared == nullptr || raw_probe == nullptr ||
      VulkanKernelTemplateKindOf(prepared) !=
          VulkanKernelTemplateKind::MapRecurrence) {
    return false;
  }
  const auto *const cached =
      static_cast<const VulkanMapRecurrenceTemplate *>(prepared);
  const auto *const probe =
      static_cast<const VulkanRecurrenceTemplateProbe *>(raw_probe);
  const MapRecurrence *const recurrence = probe->recurrence;
  if (cached->signature == nullptr || cached->prepared == nullptr ||
      cached->descriptors == nullptr || probe->signature == nullptr ||
      probe->plan == nullptr || probe->adapter == nullptr ||
      cached->signature->pick == nullptr ||
      cached->signature->pick != probe->signature->pick ||
      cached->signature->ops == nullptr ||
      cached->signature->ops != probe->signature->ops ||
      cached->signature->steps == nullptr ||
      cached->signature->step_count != 1u ||
      probe->signature->steps == nullptr ||
      probe->signature->step_count != 1u || recurrence == nullptr ||
      recurrence->first == nullptr ||
      recurrence->canonical_artifact == nullptr ||
      cached->history != probe->history ||
      probe->history != (recurrence->history != nullptr) ||
      cached->signature->steps[0u].step != recurrence->first->step ||
      probe->signature->steps[0u].step != recurrence->first->step ||
      !VulkanMapTemplateMatches(*cached->prepared, *probe->adapter,
                                probe->plan->plan, recurrence->bindings)) {
    return false;
  }
  const MapRecurrencePreparationPlan cached_plan = PlanMapRecurrencePreparation(
      *cached->signature, 1u, cached->history ? 1u : 0u);
  return SameMapRecurrenceTemplate(cached_plan, *probe->plan, probe->history);
}

struct VulkanRecurrenceTemplateVariant final {
  std::uint64_t hi{0x76756c6b2e726563ull};
  std::uint64_t lo{0x757272656e63652eull};
  bool ok{};
};

inline void MixVulkanRecurrenceVariant(std::uint64_t &hash,
                                       const std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
}

[[nodiscard]] VulkanRecurrenceTemplateVariant
VulkanRecurrenceVariant(const MapRecurrencePreparationPlan &preparation,
                        const bool history) noexcept {
  VulkanRecurrenceTemplateVariant result{};
  const std::uint64_t alignment = preparation.binding_alignment;
  const MapRecurrenceSourcePlan &source =
      history ? preparation.history_source : preparation.terminal_source;
  if (!preparation.eligible() || alignment == 0u || !source.ok ||
      source.history != history ||
      (history ? preparation.history_group_count == 0u
               : preparation.terminal_group_count() == 0u)) {
    return result;
  }
  const auto mix = [&](const std::uint64_t value) noexcept {
    MixVulkanRecurrenceVariant(result.hi, value);
    std::swap(result.hi, result.lo);
  };
  for (const std::uint64_t value :
       {static_cast<std::uint64_t>(history),
        alignment,
        preparation.window_count,
        static_cast<std::uint64_t>(preparation.input_count),
        static_cast<std::uint64_t>(preparation.output_count),
        source.exact_source_bytes,
        source.source_upper_bytes,
        source.source_storage_upper_bytes,
        source.metadata_storage_upper_bytes,
        static_cast<std::uint64_t>(source.history),
        preparation.plan.phase_id,
        preparation.plan.tile_count,
        preparation.plan.op_hash_hi,
        preparation.plan.op_hash_lo,
        static_cast<std::uint64_t>(preparation.plan.api),
        static_cast<std::uint64_t>(preparation.plan.scalar),
        static_cast<std::uint64_t>(preparation.plan.domain),
        preparation.plan.input_buffer_count,
        preparation.plan.output_buffer_count,
        preparation.plan.input_bytes_per_tile,
        preparation.plan.output_bytes_per_tile,
        preparation.plan.param_bytes,
        preparation.plan.metadata_bytes_per_tile,
        preparation.plan.bytes_per_tile,
        preparation.plan.staging_bytes,
        preparation.plan.dispatch_window_tiles,
        preparation.plan.dispatch_count,
        static_cast<std::uint64_t>(preparation.plan.fixed_authoritative),
        static_cast<std::uint64_t>(preparation.plan.ok)}) {
    mix(value);
  }
  const auto &fixed = preparation.plan.fixed_format;
  for (const std::uint64_t value :
       {static_cast<std::uint64_t>(fixed.integer_bits),
        static_cast<std::uint64_t>(fixed.fraction_bits),
        static_cast<std::uint64_t>(fixed.rounding),
        static_cast<std::uint64_t>(fixed.overflow),
        static_cast<std::uint64_t>(fixed.approximation)}) {
    mix(value);
  }
  for (const PreparedKernelProgramBindingIdentity &identity :
       preparation.input_layouts()) {
    mix(identity.offset_bytes % alignment);
    mix(identity.stride_bytes);
  }
  for (const PreparedKernelProgramBindingIdentity &identity :
       preparation.output_layouts()) {
    mix(identity.offset_bytes % alignment);
    mix(identity.stride_bytes);
    if (history) {
      std::uint64_t pitch = 0u;
      if (!rund::kernel::checked::mul(identity.count, identity.stride_bytes,
                                      pitch)) {
        return result;
      }
      mix(pitch);
    }
  }
  result.ok = true;
  return result;
}

[[nodiscard]] bool ValidCachedRecurrenceTemplate(
    const VulkanMapRecurrenceTemplate &cached, VulkanAdapter &adapter,
    const std::uint64_t group_capacity,
    const std::uint64_t descriptor_set_capacity) noexcept {
  return cached.group_capacity == group_capacity &&
         cached.descriptor_set_capacity == descriptor_set_capacity &&
         cached.prepared != nullptr && cached.prepared->adapter == &adapter &&
         cached.prepared->pipeline != nullptr &&
         cached.prepared->control_pipeline == nullptr &&
         cached.prepared->check_pipeline == nullptr &&
         cached.prepared->checks.empty() && cached.descriptors != nullptr &&
         cached.descriptors->adapter == &adapter &&
         descriptor_set_capacity <=
             static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max()) &&
         cached.descriptors->sets.size() ==
             static_cast<std::size_t>(descriptor_set_capacity);
}

} // namespace

rund::AccelCheck AcquireVulkanRecurrenceTemplate(
    PreparedKernelTemplateRegistry &registry, VulkanAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence, const std::uint64_t group_capacity,
    std::shared_ptr<VulkanMapRecurrenceTemplate> &prepared) {
  prepared.reset();
  std::uint64_t descriptor_set_capacity = 0u;
  const bool history = recurrence.history != nullptr;
  const std::span<const std::uint64_t> history_pitch_bytes =
      RecurrenceHistoryPitches(recurrence);
  if (registry.owner == nullptr || !ValidRecurrence(recurrence) ||
      signature.steps == nullptr || signature.step_count != 1u ||
      signature.steps[0u].step != recurrence.first->step ||
      group_capacity == 0u ||
      !rund::kernel::checked::mul(group_capacity, recurrence.window_count,
                                  descriptor_set_capacity) ||
      descriptor_set_capacity == 0u ||
      descriptor_set_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const MapRecurrencePreparationPlan preparation =
      PlanMapRecurrencePreparation(signature, 1u, history ? 1u : 0u);
  if (!RuntimeRecurrenceMatchesPreparedPlan(preparation, recurrence, 1u,
                                            history ? 1u : 0u,
                                            adapter.storage_align)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const VulkanRecurrenceTemplateVariant variant =
      VulkanRecurrenceVariant(preparation, history);
  if (!variant.ok) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const VulkanRecurrenceTemplateProbe probe{
      .recurrence = &recurrence,
      .signature = &signature,
      .plan = &preparation,
      .adapter = &adapter,
      .history = history,
  };
  std::shared_ptr<void> found = FindPreparedKernelTemplate(
      registry, recurrence.first->step, variant.hi, variant.lo,
      MatchVulkanRecurrenceTemplate, &probe);
  if (found != nullptr) {
    auto cached = std::static_pointer_cast<VulkanMapRecurrenceTemplate>(found);
    if (cached == nullptr ||
        !ValidCachedRecurrenceTemplate(*cached, adapter, group_capacity,
                                       descriptor_set_capacity)) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    prepared = std::move(cached);
    return rund::AccelCheck{true, "ok"};
  }

  std::uint64_t final_source_upper = 0u;
  rund::kernel::LoweringArtifact artifact{};
  if (!MapSpecializedSourceUpperBytes(recurrence.source_plan.exact_source_bytes,
                                      recurrence.source_plan.source_upper_bytes,
                                      recurrence.plan, final_source_upper) ||
      !MaterializeMapRecurrenceArtifact(
          *recurrence.canonical_artifact, recurrence.source_plan,
          recurrence.plan.input_buffer_count,
          recurrence.plan.output_buffer_count, history_pitch_bytes, artifact,
          final_source_upper)) {
    return rund::AccelCheck{false,
                            "compute_pipeline_recurrence_source_invalid"};
  }
  std::shared_ptr<const VulkanMapTemplateResources> native;
  const rund::AccelCheck ready = PrepareVulkanMapOwnedTemplate(
      pick, recurrence.plan, std::move(artifact), recurrence.windows,
      recurrence.window_count, recurrence.bindings, recurrence.first->control,
      native);
  if (!ready.ok || native == nullptr || native->pipeline == nullptr ||
      native->control_pipeline != nullptr ||
      native->check_pipeline != nullptr || !native->checks.empty()) {
    return ready.ok ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
                    : ready;
  }
  std::shared_ptr<VulkanMapDescriptorArena> descriptors;
  if (!PrepareVulkanMapDescriptorArena(adapter, *native->pipeline,
                                       descriptor_set_capacity, descriptors)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }

  auto owner = std::make_shared<VulkanMapRecurrenceTemplate>();
  owner->signature = &signature;
  owner->group_capacity = group_capacity;
  owner->descriptor_set_capacity = descriptor_set_capacity;
  owner->prepared = std::move(native);
  owner->descriptors = std::move(descriptors);
  owner->history = history;
  std::shared_ptr<void> published = owner;
  if (signature.ops == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  const rund::AccelCheck stored = PublishPreparedKernelTemplate(
      registry, recurrence.first->step, variant.hi, variant.lo, *signature.ops,
      MatchVulkanRecurrenceTemplate, &probe, published);
  if (!stored.ok || published == nullptr ||
      !MatchVulkanRecurrenceTemplate(published.get(), &probe)) {
    return stored.ok ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
                     : stored;
  }
  auto cached =
      std::static_pointer_cast<VulkanMapRecurrenceTemplate>(published);
  if (cached == nullptr ||
      !ValidCachedRecurrenceTemplate(*cached, adapter, group_capacity,
                                     descriptor_set_capacity)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  prepared = std::move(cached);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
