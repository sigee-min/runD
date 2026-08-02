#include "recurrence.hpp"

#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/recurrence/plan.hpp"
#include "../../../kernel/recurrence/source.hpp"
#include "../../../kernel/step/map/stride.hpp"
#include "../../buffer/create/telemetry.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] constexpr bool
SameRecurrenceSourcePlan(const MapRecurrenceSourcePlan &left,
                         const MapRecurrenceSourcePlan &right) noexcept {
  return left.exact_source_bytes == right.exact_source_bytes &&
         left.source_upper_bytes == right.source_upper_bytes &&
         left.source_storage_upper_bytes == right.source_storage_upper_bytes &&
         left.metadata_storage_upper_bytes ==
             right.metadata_storage_upper_bytes &&
         left.history == right.history && left.ok == right.ok;
}

[[nodiscard]] std::span<const std::uint64_t>
RecurrenceHistoryPitches(const MapRecurrence &recurrence) noexcept {
  return recurrence.history == nullptr ? std::span<const std::uint64_t>{}
                                       : recurrence.history->pitches();
}

[[nodiscard]] bool ValidRecurrence(const MapRecurrence &recurrence) noexcept {
  const std::span<const std::uint64_t> pitches =
      RecurrenceHistoryPitches(recurrence);
  return recurrence.ready() && recurrence.first != nullptr &&
         recurrence.last != nullptr && recurrence.first->step != nullptr &&
         recurrence.canonical_artifact != nullptr &&
         recurrence.canonical_artifact == &recurrence.first->step->artifact &&
         recurrence.canonical_artifact->ok && recurrence.source_plan.ok &&
         recurrence.source_plan.history == !pitches.empty() &&
         (!recurrence.source_plan.history ||
          pitches.size() == recurrence.plan.output_buffer_count) &&
         recurrence.plan.ok &&
         recurrence.plan.api == rund::kernel::ComputeApi::Vulkan &&
         recurrence.plan.input_buffer_count ==
             recurrence.bindings.resident_inputs.count &&
         recurrence.plan.output_buffer_count ==
             recurrence.bindings.resident_outputs.count &&
         recurrence.bindings.resident_inputs.has_refs() &&
         recurrence.bindings.resident_inputs.has_handles() &&
         recurrence.bindings.resident_outputs.has_refs() &&
         recurrence.bindings.resident_outputs.has_handles() &&
         recurrence.windows != nullptr && recurrence.window_count != 0u &&
         recurrence.window_count == recurrence.plan.dispatch_count &&
         recurrence.iterations >= 2u &&
         recurrence.iterations <= std::numeric_limits<std::uint32_t>::max() &&
         !recurrence.first->control.active();
}

[[nodiscard]] bool
SameIdentityLayout(const PreparedKernelProgramBindingIdentity &identity,
                   const rund::kernel::ResidentBufferRef *const ref,
                   const std::uint64_t alignment,
                   const std::uint64_t count_divisor) noexcept {
  return ref != nullptr && alignment != 0u && count_divisor != 0u &&
         ref->count % count_divisor == 0u &&
         identity.offset_bytes % alignment == ref->offset_bytes % alignment &&
         identity.element_bytes == ref->element_bytes &&
         identity.stride_bytes == ref->stride_bytes &&
         identity.count == ref->count / count_divisor &&
         identity.usage == ref->usage;
}

[[nodiscard]] bool RuntimeRecurrenceMatchesPreparedPlan(
    const MapRecurrencePreparationPlan &planned,
    const MapRecurrence &recurrence, const std::uint64_t route_group_count,
    const std::uint64_t route_history_group_count,
    const std::uint64_t alignment) noexcept {
  if (!planned.eligible() || planned.authority != recurrence.first->step ||
      planned.canonical_artifact != recurrence.canonical_artifact ||
      planned.group_count != route_group_count ||
      planned.history_group_count != route_history_group_count ||
      planned.binding_alignment != alignment ||
      planned.window_count != recurrence.window_count ||
      !backend_template_plan::same_plan(planned.plan, recurrence.plan) ||
      planned.input_count != recurrence.bindings.resident_inputs.count ||
      planned.output_count != recurrence.bindings.resident_outputs.count) {
    return false;
  }
  const MapRecurrenceSourcePlan &expected_source = recurrence.history == nullptr
                                                       ? planned.terminal_source
                                                       : planned.history_source;
  if (!SameRecurrenceSourcePlan(expected_source, recurrence.source_plan)) {
    return false;
  }
  for (std::size_t index = 0u; index < planned.input_count; ++index) {
    if (!SameIdentityLayout(planned.inputs[index],
                            recurrence.bindings.resident_inputs.ref(index),
                            alignment, 1u)) {
      return false;
    }
  }
  const std::uint64_t output_divisor =
      recurrence.history == nullptr ? 1u : recurrence.iterations;
  for (std::size_t index = 0u; index < planned.output_count; ++index) {
    if (!SameIdentityLayout(planned.outputs[index],
                            recurrence.bindings.resident_outputs.ref(index),
                            alignment, output_divisor)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
RuntimeRecurrenceMatchesPlan(const BackendRun &owner,
                             const MapRecurrence &recurrence,
                             const std::uint64_t route_group_count,
                             const std::uint64_t route_history_group_count,
                             const std::uint64_t alignment) noexcept {
  const MapRecurrencePreparationPlan planned = PlanMapRecurrencePreparation(
      owner, route_group_count, route_history_group_count);
  return RuntimeRecurrenceMatchesPreparedPlan(
      planned, recurrence, route_group_count, route_history_group_count,
      alignment);
}

[[nodiscard]] bool SameRuntimeRecurrenceTemplate(
    const BackendRun &left_owner, const MapRecurrence &left,
    const BackendRun &right_owner, const MapRecurrence &right,
    const std::uint64_t alignment) noexcept {
  const bool history = left.history != nullptr;
  if (left_owner.pick == nullptr || left_owner.pick != right_owner.pick ||
      left_owner.ops == nullptr || left_owner.ops != right_owner.ops ||
      history != (right.history != nullptr)) {
    return false;
  }
  const MapRecurrencePreparationPlan left_plan =
      PlanMapRecurrencePreparation(left_owner, 1u, history ? 1u : 0u);
  const MapRecurrencePreparationPlan right_plan =
      PlanMapRecurrencePreparation(right_owner, 1u, history ? 1u : 0u);
  return left_plan.binding_alignment == alignment &&
         right_plan.binding_alignment == alignment &&
         SameMapRecurrenceTemplate(left_plan, right_plan, history);
}

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

[[nodiscard]] rund::AccelCheck AcquireVulkanRecurrenceTemplate(
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

[[nodiscard]] rund::AccelCheck
PrepareRecurrenceMap(PreparedKernelTemplateRegistry &registry,
                     VulkanAdapter &adapter, const rund::AccelDevice &pick,
                     const BackendRun &owner, const MapRecurrence &recurrence,
                     const std::uint64_t group_capacity,
                     std::shared_ptr<void> &resource) {
  std::shared_ptr<VulkanMapRecurrenceTemplate> recurrence_template;
  const rund::AccelCheck acquired = AcquireVulkanRecurrenceTemplate(
      registry, adapter, pick, owner, recurrence, group_capacity,
      recurrence_template);
  if (!acquired.ok || recurrence_template == nullptr) {
    return acquired.ok
               ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
               : acquired;
  }
  const rund::AccelCheck ready = PrepareVulkanMapProvedRoute(
      pick, recurrence.plan, recurrence.windows, recurrence.window_count,
      recurrence.bindings, recurrence.first->control,
      recurrence.history != nullptr, recurrence_template->prepared,
      recurrence_template->descriptors, resource,
      static_cast<std::uint32_t>(recurrence.iterations));
  auto *const map = static_cast<VulkanMapEncodeResources *>(resource.get());
  if (!ready.ok || map == nullptr || map->adapter != &adapter ||
      map->prepared == nullptr || map->controlled() || map->windows.empty() ||
      map->prepared->plan.dispatch_count != map->windows.size()) {
    return ready.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                    : ready;
  }
  map->binding_owner = recurrence.history;
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] bool
ValidRecurrenceReservation(const PreparedKernelTemplateRegistry &registry,
                           const PreparedPipelineStatusLayout &status,
                           const std::uint64_t group_count,
                           const std::uint64_t history_group_count,
                           const std::uint64_t terminal_template_group_capacity,
                           const std::uint64_t history_template_group_capacity,
                           const std::uint64_t template_count) noexcept {
  if (group_count == 0u || history_group_count > group_count ||
      (status.generation_stride != 1u && status.generation_stride != 2u) ||
      !registry.limit.ok) {
    return false;
  }
  std::uint64_t expected_groups = 0u;
  std::uint64_t expected_history = 0u;
  std::uint64_t expected_terminal = 0u;
  if (!rund::kernel::checked::mul(group_count, status.generation_stride,
                                  expected_groups) ||
      !rund::kernel::checked::mul(history_group_count, status.generation_stride,
                                  expected_history) ||
      expected_history > expected_groups) {
    return false;
  }
  expected_terminal = expected_groups - expected_history;
  const PreparedMapRecurrenceReservation &limit = registry.limit.map_recurrence;
  const PreparedMapRecurrenceReservation &consumed =
      registry.reservation.map_recurrence;
  if (limit.group_count != expected_groups ||
      limit.history_group_count != expected_history ||
      limit.terminal_template_group_capacity != expected_terminal ||
      limit.history_template_group_capacity != expected_history ||
      limit.template_count != template_count ||
      terminal_template_group_capacity != expected_terminal ||
      history_template_group_capacity != expected_history ||
      limit.route_step_count != expected_groups ||
      consumed.template_count != limit.template_count ||
      consumed.terminal_template_group_capacity !=
          limit.terminal_template_group_capacity ||
      consumed.history_template_group_capacity !=
          limit.history_template_group_capacity ||
      consumed.group_count < group_count ||
      consumed.group_count > expected_groups ||
      consumed.group_count % group_count != 0u) {
    return false;
  }
  const std::uint64_t consumed_streams = consumed.group_count / group_count;
  std::uint64_t consumed_history = 0u;
  return consumed_streams != 0u &&
         consumed_streams <= status.generation_stride &&
         rund::kernel::checked::mul(history_group_count, consumed_streams,
                                    consumed_history) &&
         consumed.history_group_count == consumed_history &&
         consumed.route_step_count == consumed.group_count;
}

} // namespace

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    const std::span<const BackendBatchEntry> entries,
    const MapRecurrence &recurrence, PreparedKernelTemplateRegistry &registry,
    PreparedPipelineStatusLayout &status, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory) {
  if (!ValidRecurrence(recurrence) || entries.empty() ||
      !ValidRecurrenceReservation(
          registry, status, 1u, recurrence.history == nullptr ? 0u : 1u,
          recurrence.history == nullptr ? status.generation_stride : 0u,
          recurrence.history == nullptr ? 0u : status.generation_stride, 1u)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    const BackendBatchEntry &entry = entries[index];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        entry.run == nullptr || entry.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*entry.run->pick, context);
    if (!valid.ok || context.adapter == nullptr ||
        (pipeline.adapter != nullptr && pipeline.adapter != context.adapter) ||
        !SetPreparedProgramStatusSlice(status,
                                       static_cast<std::uint32_t>(index), 0u)) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }
    pipeline.adapter = context.adapter;
  }
  const BackendRun *const owner = entries.front().run;
  if (pipeline.adapter == nullptr || owner == nullptr ||
      owner->pick == nullptr ||
      !RuntimeRecurrenceMatchesPlan(*owner, recurrence, 1u,
                                    recurrence.history == nullptr ? 0u : 1u,
                                    pipeline.adapter->storage_align)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t group_capacity = status.generation_stride;
  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  rund::AccelCheck ready{};
  try {
    ready =
        PrepareRecurrenceMap(registry, *pipeline.adapter, *owner->pick, *owner,
                             recurrence, group_capacity, pipeline.recurrence);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!ready.ok) {
    return ready;
  }
  pipeline.dispatch_count = recurrence.window_count;
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
PrepareVulkanTransducers(const std::span<const BackendBatchEntry> templates,
                         const std::span<const TileTransducer> transducers,
                         PreparedKernelTemplateRegistry &registry,
                         const PreparedPipelineStatusLayout &status,
                         VulkanPipeline &pipeline,
                         PreparedMemory &staging_memory) {
  staging_memory = {};
  pipeline.transducers.clear();
  if (transducers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (pipeline.adapter == nullptr ||
      transducers.size() > PreparedPipelineStepCapacity) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }

  struct RouteDemand final {
    const BackendRun *owner{};
    std::uint64_t group_count{};
    std::uint64_t history_group_count{};
  };
  struct TemplateDemand final {
    std::size_t representative{};
    std::uint64_t group_count{};
    std::uint64_t group_capacity{};
  };
  std::array<RouteDemand, PreparedPipelineStepCapacity> route_demands{};
  std::array<TemplateDemand, PreparedPipelineStepCapacity> template_demands{};
  std::array<std::size_t, PreparedPipelineStepCapacity> route_demand_indices{};
  std::array<std::size_t, PreparedPipelineStepCapacity>
      template_demand_indices{};
  std::size_t route_demand_count = 0u;
  std::size_t template_demand_count = 0u;
  std::uint64_t history_group_count = 0u;

  for (std::size_t index = 0u; index < transducers.size(); ++index) {
    const TileTransducer &transducer = transducers[index];
    const MapRecurrence &recurrence = transducer.recurrence;
    if (!ValidRecurrence(recurrence) ||
        recurrence.iterations != transducer.template_count ||
        transducer.template_first >= templates.size() ||
        transducer.template_count >
            templates.size() - transducer.template_first) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BackendBatchEntry &owner = templates[transducer.template_first];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        owner.run == nullptr || owner.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*owner.run->pick, context);
    if (!valid.ok || context.adapter != pipeline.adapter) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }

    std::size_t route_demand = 0u;
    while (route_demand < route_demand_count &&
           route_demands[route_demand].owner != owner.run) {
      ++route_demand;
    }
    if (route_demand == route_demand_count) {
      route_demands[route_demand] = RouteDemand{.owner = owner.run};
      ++route_demand_count;
    }
    RouteDemand &route = route_demands[route_demand];
    if (!rund::kernel::checked::add(route.group_count, 1u, route.group_count) ||
        (recurrence.history != nullptr &&
         (!rund::kernel::checked::add(route.history_group_count, 1u,
                                      route.history_group_count) ||
          !rund::kernel::checked::add(history_group_count, 1u,
                                      history_group_count)))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    route_demand_indices[index] = route_demand;

    std::size_t template_demand = 0u;
    for (; template_demand < template_demand_count; ++template_demand) {
      const TileTransducer &representative =
          transducers[template_demands[template_demand].representative];
      const BackendBatchEntry &representative_owner =
          templates[representative.template_first];
      if (representative_owner.run != nullptr &&
          SameRuntimeRecurrenceTemplate(
              *owner.run, recurrence, *representative_owner.run,
              representative.recurrence, pipeline.adapter->storage_align)) {
        break;
      }
    }
    if (template_demand == template_demand_count) {
      template_demands[template_demand] =
          TemplateDemand{.representative = index};
      ++template_demand_count;
    }
    if (!rund::kernel::checked::add(
            template_demands[template_demand].group_count, 1u,
            template_demands[template_demand].group_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    template_demand_indices[index] = template_demand;
  }

  for (std::size_t index = 0u; index < transducers.size(); ++index) {
    const TileTransducer &transducer = transducers[index];
    const BackendBatchEntry &owner = templates[transducer.template_first];
    const RouteDemand &route = route_demands[route_demand_indices[index]];
    if (route.owner != owner.run || route.group_count == 0u ||
        !RuntimeRecurrenceMatchesPlan(
            *owner.run, transducer.recurrence, route.group_count,
            route.history_group_count, pipeline.adapter->storage_align)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }

  std::uint64_t terminal_template_group_capacity = 0u;
  std::uint64_t history_template_group_capacity = 0u;
  for (std::size_t index = 0u; index < template_demand_count; ++index) {
    TemplateDemand &demand = template_demands[index];
    if (demand.group_count == 0u ||
        !rund::kernel::checked::mul(demand.group_count,
                                    status.generation_stride,
                                    demand.group_capacity)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const MapRecurrence &representative =
        transducers[demand.representative].recurrence;
    std::uint64_t &variant_capacity = representative.history == nullptr
                                          ? terminal_template_group_capacity
                                          : history_template_group_capacity;
    if (!rund::kernel::checked::add(variant_capacity, demand.group_capacity,
                                    variant_capacity)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!ValidRecurrenceReservation(
          registry, status, transducers.size(), history_group_count,
          terminal_template_group_capacity, history_template_group_capacity,
          template_demand_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  try {
    pipeline.transducers.resize(transducers.size());
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  try {
    for (std::size_t index = 0u; index < transducers.size(); ++index) {
      const TileTransducer &transducer = transducers[index];
      const BackendBatchEntry &owner = templates[transducer.template_first];
      const std::uint64_t group_capacity =
          template_demands[template_demand_indices[index]].group_capacity;
      if (group_capacity == 0u) {
        pipeline.transducers.clear();
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const rund::AccelCheck ready = PrepareRecurrenceMap(
          registry, *pipeline.adapter, *owner.run->pick, *owner.run,
          transducer.recurrence, group_capacity, pipeline.transducers[index]);
      if (!ready.ok) {
        pipeline.transducers.clear();
        return ready;
      }
    }
  } catch (const std::bad_alloc &) {
    pipeline.transducers.clear();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    pipeline.transducers.clear();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
