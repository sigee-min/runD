#include "../build.hpp"

#include "../aggregate/admit.hpp"

#include "../../../buffer/resident/find.hpp"
#include "../../../resident.hpp"
#include "../../../resident/access.hpp"
#include "../../../runtime/map/api.hpp"
#include "../../../runtime/map/resources.hpp"

#include "../../../../kernel/recurrence/plan.hpp"
#include "../../../../kernel/recurrence/source.hpp"
#include "../../../pipeline/guard.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::span<const std::uint64_t>
RecurrenceHistoryPitches(const MapRecurrence &recurrence) noexcept {
  return recurrence.history == nullptr ? std::span<const std::uint64_t>{}
                                       : recurrence.history->pitches();
}

struct MetalRecurrenceTemplateProbe final {
  const MapRecurrencePreparationPlan *plan{};
  MetalAdapter *adapter{};
  bool history{};
};

[[nodiscard]] bool
MatchMetalRecurrenceTemplate(const void *const prepared,
                             const void *const raw_probe) noexcept {
  if (prepared == nullptr || raw_probe == nullptr ||
      MetalKernelTemplateKindOf(prepared) !=
          MetalKernelTemplateKind::MapRecurrence) {
    return false;
  }
  const auto *const cached =
      static_cast<const MetalMapRecurrenceTemplate *>(prepared);
  const auto *const probe =
      static_cast<const MetalRecurrenceTemplateProbe *>(raw_probe);
  if (cached->signature == nullptr || cached->prepared == nullptr ||
      cached->prepared->adapter != probe->adapter || probe->plan == nullptr ||
      probe->adapter == nullptr || cached->history != probe->history) {
    return false;
  }
  const MapRecurrencePreparationPlan cached_plan = PlanMapRecurrencePreparation(
      *cached->signature, 1u, cached->history ? 1u : 0u);
  return SameMapRecurrenceTemplate(cached_plan, *probe->plan, probe->history);
}

inline constexpr std::uint64_t MetalRecurrenceVariantHi = 0x6d6574616c2e7265ull;
inline constexpr std::uint64_t MetalTerminalRecurrenceVariantLo =
    0x6375722e7465726dull;
inline constexpr std::uint64_t MetalHistoryRecurrenceVariantLo =
    0x6375722e68697374ull;

[[nodiscard]] rund::AccelCheck AcquireMetalRecurrenceTemplate(
    PreparedKernelTemplateRegistry &registry, MetalAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
  prepared.reset();
  const std::span<const std::uint64_t> history_pitch_bytes =
      RecurrenceHistoryPitches(recurrence);
  const bool history = !history_pitch_bytes.empty();
  const MapRecurrencePreparationPlan normalized =
      PlanMapRecurrencePreparation(signature, 1u, history ? 1u : 0u);
  const MapRecurrenceSourcePlan &source_plan =
      history ? normalized.history_source : normalized.terminal_source;
  if (registry.owner == nullptr || recurrence.first == nullptr ||
      recurrence.first->step == nullptr || signature.steps == nullptr ||
      signature.ops == nullptr || signature.step_count != 1u ||
      signature.steps[0u].step != recurrence.first->step ||
      !normalized.eligible() || normalized.group_count != 1u ||
      normalized.history_group_count != (history ? 1u : 0u) ||
      normalized.binding_alignment != 1u ||
      normalized.authority != recurrence.first->step ||
      normalized.canonical_artifact == nullptr || !source_plan.ok ||
      source_plan.history != history ||
      (history && history_pitch_bytes.size() != normalized.output_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::uint64_t variant_lo = history ? MetalHistoryRecurrenceVariantLo
                                           : MetalTerminalRecurrenceVariantLo;
  const MetalRecurrenceTemplateProbe probe{
      .plan = &normalized,
      .adapter = &adapter,
      .history = history,
  };
  std::shared_ptr<void> found = FindPreparedKernelTemplate(
      registry, normalized.authority, MetalRecurrenceVariantHi, variant_lo,
      MatchMetalRecurrenceTemplate, &probe);
  if (found != nullptr) {
    const auto cached =
        std::static_pointer_cast<MetalMapRecurrenceTemplate>(found);
    if (cached == nullptr || cached->prepared == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    prepared = cached->prepared;
    return rund::AccelCheck{true, "ok"};
  }

  // A miss owns exactly one transformed source. PrepareMetalMapTemplate moves
  // the specialized/guarded text into the adapter cache; this local artifact
  // dies before any route resource is allocated. Reserve the final guarded
  // upper during initial recurrence emission so every later edit is in-place.
  std::uint64_t specialized_upper = 0u;
  std::uint64_t final_source_upper = 0u;
  if (recurrence.first->control.active() ||
      !normalized.canonical_artifact->metadata.read_routes.empty() ||
      !MapSpecializedSourceUpperBytes(source_plan.exact_source_bytes,
                                      source_plan.source_upper_bytes,
                                      normalized.plan, specialized_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(specialized_upper, 1u, true,
                                            final_source_upper)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  rund::kernel::LoweringArtifact artifact{};
  if (!MaterializeMapRecurrenceArtifact(
          *normalized.canonical_artifact, source_plan, normalized.input_count,
          normalized.output_count, history_pitch_bytes, artifact,
          final_source_upper)) {
    return rund::AccelCheck{false,
                            "compute_pipeline_recurrence_source_invalid"};
  }
  std::shared_ptr<const MetalMapTemplateResources> native;
  const rund::AccelCheck ready = PrepareMetalMapOwnedTemplate(
      pick, normalized.plan, std::move(artifact), recurrence.windows,
      recurrence.window_count, recurrence.bindings, recurrence.first->control,
      native);
  if (!ready.ok) {
    return ready;
  }

  auto owner = std::make_shared<MetalMapRecurrenceTemplate>();
  owner->signature = &signature;
  owner->history = history;
  owner->prepared = std::move(native);
  std::shared_ptr<void> published = owner;
  const rund::AccelCheck stored = PublishPreparedKernelTemplate(
      registry, normalized.authority, MetalRecurrenceVariantHi, variant_lo,
      *signature.ops, MatchMetalRecurrenceTemplate, &probe, published);
  if (!stored.ok || published == nullptr ||
      !MatchMetalRecurrenceTemplate(published.get(), &probe)) {
    return stored.ok ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
                     : stored;
  }
  const auto cached =
      std::static_pointer_cast<MetalMapRecurrenceTemplate>(published);
  if (cached == nullptr || cached->prepared == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  prepared = cached->prepared;
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck PrepareMetalRecurrenceRoute(
    PreparedKernelTemplateRegistry &registry, MetalAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence, const BoundControl &control,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  std::shared_ptr<const MetalMapTemplateResources> prepared;
  const rund::AccelCheck acquired = AcquireMetalRecurrenceTemplate(
      registry, adapter, pick, signature, recurrence, prepared);
  return acquired.ok ? PrepareMetalMapProvedRoute(
                           pick, recurrence.plan, recurrence.windows,
                           recurrence.window_count, recurrence.bindings,
                           control, std::move(prepared), resources, iterations)
                     : acquired;
}

} // namespace

rund::AccelCheck MetalPipelineBuild::Admit() {
  if (templates.empty() || templates.size() != status.active_step_count ||
      templates.front().run == nullptr ||
      templates.front().run->pick == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const bool compact_input = entries.empty() && barriers.empty() &&
                             status.command_count == 2u &&
                             aggregates.size() == 1u;
  if (!compact_input) {
    if (entries.empty() || entries.size() != barriers.size() ||
        entries.size() != status.command_count ||
        entries.front().run == nullptr ||
        entries.front().run->pick == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    for (std::size_t index = 0u; index < entries.size(); ++index) {
      failure_context.occurrence_route(entries[index]);
      if (entries[index].occurrence_index != index ||
          entries[index].template_index >= templates.size() ||
          (entries[index].transducer != NoTileTransducer &&
           entries[index].transducer >= transducers.size())) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
    }
  }
  failure_context.clear_route();
  const rund::AccelCheck valid =
      ValidateMetalKernelContext(*templates.front().run->pick, context);
  if (!valid.ok || context.adapter == nullptr) {
    return valid;
  }
  if (aggregates.size() == 1u) {
    const NestedAggregate &aggregate = aggregates.front();
    NestedTemplateShape expected_shape{};
    const bool complete_templates =
        aggregate.shape.valid() && aggregate.shape.first() == 0u &&
        aggregate.shape.end() == templates.size() &&
        ProveNestedTemplateShape(aggregate.shape.first(), aggregate.maximum,
                                 aggregate.tile, aggregate.shape.inner_bound(),
                                 expected_shape) &&
        expected_shape == aggregate.shape;
    const bool complete_publication =
        publications.size() == 1u && aggregate.publication_index == 0u;
    const bool profile_ready =
        !profile_steps || aggregate.profile.aggregate_profile_supported;
    const bool seed_profile_owner =
        aggregate.shape.seed_first() < status.active_step_count &&
        status.declared_steps[aggregate.shape.seed_first()] <
            status.declared_step_count;
    if (complete_templates && complete_publication && profile_ready &&
        seed_profile_owner) {
      const rund::AccelCheck admitted = AdmitMetalNestedAggregate(
          aggregate, status, profile_steps, context, native_aggregate);
      if (!admitted.ok) {
        if (compact_input || std::string_view{admitted.reason} !=
                                 "accel_kernel_primitive_unsupported") {
          return admitted;
        }
      } else {
        aggregate_profile_owner =
            status.declared_steps[aggregate.shape.seed_first()];
        aggregate_selected = true;
      }
    }
  }
  if (aggregate_selected) {
    try {
      pipeline = std::make_shared<MetalSequence>();
      pipeline->adapter = context.adapter;
      pipeline->profile_steps = profile_steps;
      if (profile_steps) {
        if (status.declared_step_count == 0u ||
            status.declared_step_count > PreparedPipelineStepCapacity) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        pipeline->step_evidence.resize(status.declared_step_count);
      }
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    } catch (const std::length_error &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    return rund::AccelCheck{true, "ok"};
  }
  if (compact_input) {
    // The compact representation intentionally owns no canonical occurrence
    // stream. Reaching this point means the common proof and native admission
    // disagreed, so fail closed instead of manufacturing a second authority.
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  recurrence = BuildMapRecurrence(entries, barriers);
  if (recurrence.invalid()) {
    return rund::AccelCheck{false, recurrence.reason};
  }
  std::uint32_t state_count = 0u;
  try {
    constexpr std::uint32_t unset = std::numeric_limits<std::uint32_t>::max();
    struct DeferredInnerState final {
      std::uint32_t advance{unset};
      std::uint32_t bound{unset};
      bool used{};
    };
    // Window-state identity is frozen by the common Pipeline route-capacity
    // contract. Keep this admission proof on the stack: growing a heap table
    // from a backend-provided state id would create an unplanned cold owner
    // and allow a malformed id to select an allocation size.
    std::array<DeferredInnerState, PreparedPipelineStepCapacity>
        deferred_inner{};
    std::size_t planned_native_window_count = 0u;
    for (const BackendBatchEntry &entry : entries) {
      failure_context.occurrence_route(entry);
      const BackendWindow *const window = entry.recurrence.window;
      if (window == nullptr) {
        continue;
      }
      ++planned_native_window_count;
      if (!window->nested()) {
        continue;
      }
      if (window->state == unset || window->state >= deferred_inner.size()) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (window->phase != BackendWindowPhase::NestedFold) {
        continue;
      }
      DeferredInnerState &deferred = deferred_inner[window->state];
      if (deferred.used && (deferred.advance != window->inner_advance ||
                            deferred.bound != window->inner_bound)) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      deferred.advance = window->inner_advance;
      deferred.bound = window->inner_bound;
      deferred.used = true;
    }
    native_windows.reserve(planned_native_window_count);
    if (native_windows.capacity() != planned_native_window_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    MetalResidentState &resident = MetalResidents(*context.adapter);
    std::lock_guard resident_lock{resident.mutex};
    failure_context.clear_route();
    for (const BackendPublish &publication : publications) {
      const PreparedKernelPublicationIdentity &identity = publication.identity;
      const bool window =
          identity.kind == PreparedKernelPublicationKind::Window;
      const MetalResidentBufferResult target = ResolveMetalResidentBuffer(
          resident, publication.target.source, publication.target.handle,
          "accel_metal_resident_id_unavailable", true);
      std::array<MetalResidentBufferResult, 3u> sources{};
      MetalResidentBufferResult count{};
      if (window) {
        count = ResolveMetalResidentBuffer(
            resident, publication.count.source, publication.count.handle,
            "accel_metal_resident_id_unavailable", true);
      }
      if (!target.check.ok || target.device_buffer == nullptr ||
          identity.state == std::numeric_limits<std::uint32_t>::max() ||
          identity.state >= deferred_inner.size() ||
          (!window && identity.final >= publication.sources.size()) ||
          (window &&
           (!count.check.ok || count.device_buffer == nullptr ||
            publication.count.source.count != 1u ||
            publication.count.source.element_bytes != sizeof(std::uint32_t)))) {
        return target.check.ok
                   ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                   : target.check;
      }
      for (std::size_t bank = 0u; bank < sources.size(); ++bank) {
        const BackendRead &source = publication.sources[bank];
        sources[bank] = ResolveMetalResidentBuffer(
            resident, source.source, source.handle,
            "accel_metal_resident_id_unavailable", true);
        const bool source_valid =
            sources[bank].check.ok && sources[bank].device_buffer != nullptr &&
            ((!window && bank != identity.final) ||
             sources[bank].device_buffer != target.device_buffer) &&
            source.source.count ==
                (window ? identity.tile : publication.target.source.count) &&
            source.source.element_bytes ==
                publication.target.source.element_bytes &&
            (source.source.element_bytes == 4u ||
             source.source.element_bytes == 8u) &&
            source.source.stride_bytes >= source.source.element_bytes &&
            (source.source.offset_bytes % sizeof(std::uint32_t)) == 0u &&
            (source.source.stride_bytes % sizeof(std::uint32_t)) == 0u;
        if (!source_valid) {
          return sources[bank].check.ok
                     ? rund::AccelCheck{false,
                                        "compute_resident_stride_invalid"}
                     : sources[bank].check;
        }
      }
      state_count = std::max(state_count, identity.state + 1u);
      if (native_publication_count == native_publications.size()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      native_publications[native_publication_count++] = MetalPublish{
          .sources = {sources[0].device_buffer, sources[1].device_buffer,
                      sources[2].device_buffer},
          .target = target.device_buffer,
          .count = window ? count.device_buffer : nullptr,
          .params =
              MetalPublishParams{
                  .count = publication.target.source.count,
                  .source_offset_words =
                      {publication.sources[0].source.offset_bytes /
                           sizeof(std::uint32_t),
                       publication.sources[1].source.offset_bytes /
                           sizeof(std::uint32_t),
                       publication.sources[2].source.offset_bytes /
                           sizeof(std::uint32_t)},
                  .source_stride_words =
                      {publication.sources[0].source.stride_bytes /
                           sizeof(std::uint32_t),
                       publication.sources[1].source.stride_bytes /
                           sizeof(std::uint32_t),
                       publication.sources[2].source.stride_bytes /
                           sizeof(std::uint32_t)},
                  .target_offset_words =
                      publication.target.source.offset_bytes /
                      sizeof(std::uint32_t),
                  .target_stride_words =
                      publication.target.source.stride_bytes /
                      sizeof(std::uint32_t),
                  .element_words = static_cast<std::uint32_t>(
                      publication.target.source.element_bytes /
                      sizeof(std::uint32_t)),
                  .declared_step_count = status.declared_step_count,
                  .state = identity.state,
                  .final = identity.final,
                  .maximum = identity.maximum,
                  .tile = identity.tile,
                  .kind = static_cast<std::uint32_t>(identity.kind),
                  .count_offset_words =
                      window ? publication.count.source.offset_bytes /
                                   sizeof(std::uint32_t)
                             : 0u,
              },
      };
    }
    for (std::size_t entry_index = 0u; entry_index < entries.size();
         ++entry_index) {
      failure_context.occurrence_route(entries[entry_index]);
      const BackendWindow *const window =
          entries[entry_index].recurrence.window;
      if (window == nullptr) {
        continue;
      }
      const bool nested = window->nested();
      if (window->state == unset || window->state >= deferred_inner.size()) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      const DeferredInnerState &deferred = deferred_inner[window->state];
      const bool occurrence_valid =
          window->valid_occurrence(entries[entry_index].transduced_action());
      const bool deferred_valid =
          !nested ||
          (deferred.used && deferred.bound == window->inner_bound &&
           (deferred.advance == 0u || deferred.advance == deferred.bound));
      if (entries[entry_index].transducer != NoTileTransducer) {
        const TileTransducer &transducer =
            transducers[entries[entry_index].transducer];
        if (!occurrence_valid || !transducer.recurrence.ready() ||
            transducer.template_first != entries[entry_index].template_index ||
            transducer.template_count != window->inner_bound) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
      }
      const MetalResidentBufferResult count = ResolveMetalResidentBuffer(
          resident, window->count.source, window->count.handle,
          "accel_metal_resident_id_unavailable", true);
      if (!count.check.ok || count.device_buffer == nullptr ||
          !occurrence_valid || !deferred_valid ||
          window->count.source.count != 1u ||
          window->count.source.element_bytes != sizeof(std::uint32_t) ||
          (window->count.source.offset_bytes % sizeof(std::uint32_t)) != 0u) {
        return count.check.ok
                   ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                   : count.check;
      }
      std::array<MetalResidentBufferResult, 3u> terminals{count, count, count};
      if (window->has_terminal) {
        for (std::size_t bank = 0u; bank < terminals.size(); ++bank) {
          const BackendRead &terminal = window->terminal[bank];
          terminals[bank] = ResolveMetalResidentBuffer(
              resident, terminal.source, terminal.handle,
              "accel_metal_resident_id_unavailable", true);
          if (!terminals[bank].check.ok ||
              terminals[bank].device_buffer == nullptr ||
              terminal.source.count != 1u ||
              terminal.source.element_bytes != sizeof(std::uint32_t) ||
              (terminal.source.offset_bytes % sizeof(std::uint32_t)) != 0u) {
            return terminals[bank].check.ok
                       ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                       : terminals[bank].check;
          }
        }
      }
      state_count = std::max(state_count, window->state + 1u);
      const std::uint32_t template_index = entries[entry_index].template_index;
      if (template_index >= status.active_step_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      native_windows.push_back(MetalWindow{
          .resident = count.device_buffer,
          .terminals = {terminals[0].device_buffer, terminals[1].device_buffer,
                        terminals[2].device_buffer},
          .params =
              MetalWindowParams{
                  .count_offset_words =
                      window->count.source.offset_bytes / sizeof(std::uint32_t),
                  .terminal_offset_words =
                      {window->has_terminal
                           ? window->terminal[0].source.offset_bytes /
                                 sizeof(std::uint32_t)
                           : 0u,
                       window->has_terminal
                           ? window->terminal[1].source.offset_bytes /
                                 sizeof(std::uint32_t)
                           : 0u,
                       window->has_terminal
                           ? window->terminal[2].source.offset_bytes /
                                 sizeof(std::uint32_t)
                           : 0u},
                  .maximum = window->maximum,
                  .tile = window->tile,
                  .iteration = window->outer_iteration,
                  .expected = window->expected,
                  .state = window->state,
                  .has_terminal =
                      static_cast<std::uint32_t>(window->has_terminal),
                  .phase = static_cast<std::uint32_t>(window->phase),
                  .declared_step = status.declared_steps[template_index],
                  .overflow_reason = static_cast<std::uint32_t>(
                      rund::compute::Reason::BoundedCountInvalid),
                  .inner_bound = window->inner_bound,
                  .inner_advance =
                      window->phase == BackendWindowPhase::NestedSeed
                          ? deferred.advance
                          : window->inner_advance,
              },
          .entry = static_cast<std::uint32_t>(entry_index),
      });
    }
    if (native_windows.size() != planned_native_window_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  failure_context.clear_route();
  try {
    pipeline = std::make_shared<MetalSequence>();
    pipeline->adapter = context.adapter;
    pipeline->state_count = state_count;
    pipeline->profile_steps = profile_steps;
    pipeline->transducers.resize(transducers.size());
    if (profile_steps) {
      if (status.declared_step_count == 0u ||
          status.declared_step_count > PreparedPipelineStepCapacity) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->step_evidence.resize(status.declared_step_count);
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  failure_context.clear_route();
  if (recurrence.ready()) {
    if (!transducers.empty()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (recurrence.first == nullptr ||
        recurrence.canonical_artifact == nullptr ||
        !recurrence.source_plan.ok || recurrence.windows == nullptr ||
        recurrence.window_count == 0u || recurrence.iterations < 2u ||
        recurrence.iterations > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalRecurrenceRoute(
          template_registry, *context.adapter, *entries.front().run->pick,
          *entries.front().run, recurrence, recurrence.first->control,
          pipeline->recurrence,
          static_cast<std::uint32_t>(recurrence.iterations));
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    } catch (const std::length_error &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      return ready;
    }
    auto *const prepared =
        static_cast<MetalMapEncodeResources *>(pipeline->recurrence.get());
    if (prepared == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    prepared->binding_owner = recurrence.history;
  }
  for (std::size_t index = 0u; index < transducers.size(); ++index) {
    const TileTransducer &transducer = transducers[index];
    failure_context.template_route(transducer.template_first);
    const MapRecurrence &map = transducer.recurrence;
    if (!map.ready() || map.first == nullptr ||
        map.canonical_artifact == nullptr || !map.source_plan.ok ||
        map.windows == nullptr || map.window_count == 0u ||
        map.iterations < 2u || map.iterations != transducer.template_count ||
        map.iterations > std::numeric_limits<std::uint32_t>::max() ||
        transducer.template_first >= templates.size() ||
        transducer.template_count >
            templates.size() - transducer.template_first) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BackendBatchEntry &owner = templates[transducer.template_first];
    if (owner.run == nullptr || owner.run->pick == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalRecurrenceRoute(
          template_registry, *context.adapter, *owner.run->pick, *owner.run,
          map, map.first->control, pipeline->transducers[index],
          static_cast<std::uint32_t>(map.iterations));
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    } catch (const std::length_error &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (!ready.ok) {
      return ready;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
