#include "../build.hpp"

#include "../aggregate/admit.hpp"

#include "../../../buffer/resident/find.hpp"
#include "../../../resident.hpp"
#include "../../../resident/access.hpp"
#include "../../../runtime/map/api.hpp"
#include "../../../runtime/map/resources.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

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
      if (entries[index].occurrence_index != index ||
          entries[index].template_index >= templates.size() ||
          (entries[index].transducer != NoTileTransducer &&
           entries[index].transducer >= transducers.size())) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
    }
  }
  const rund::AccelCheck valid =
      ValidateMetalKernelContext(*templates.front().run->pick, context);
  if (!valid.ok || context.adapter == nullptr) {
    return valid;
  }
  if (aggregates.size() == 1u) {
    const NestedAggregate &aggregate = aggregates.front();
    const bool complete_templates =
        aggregate.seed.first == 0u &&
        aggregate.seed.count == aggregate.window.outer_bound &&
        aggregate.action.first == aggregate.seed.end() &&
        aggregate.action.count == aggregate.window.inner_bound &&
        aggregate.fold.first == aggregate.action.end() &&
        aggregate.fold.count == 3u && aggregate.fold.end() == templates.size();
    const bool complete_publication =
        publications.size() == 1u && aggregate.publication_index == 0u;
    const bool profile_ready =
        !profile_steps || aggregate.profile.aggregate_profile_supported;
    const bool seed_profile_owner =
        aggregate.seed.first < status.active_step_count &&
        status.declared_steps[aggregate.seed.first] <
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
        aggregate_profile_owner = status.declared_steps[aggregate.seed.first];
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
    std::vector<std::uint32_t> deferred_inner_advance;
    std::vector<std::uint32_t> deferred_inner_bound;
    for (const BackendBatchEntry &entry : entries) {
      const BackendWindow *const window = entry.recurrence.window;
      if (window == nullptr || !window->nested()) {
        continue;
      }
      if (window->state == unset) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (deferred_inner_advance.size() <= window->state) {
        deferred_inner_advance.resize(
            static_cast<std::size_t>(window->state) + 1u, unset);
        deferred_inner_bound.resize(
            static_cast<std::size_t>(window->state) + 1u, unset);
      }
      if (window->phase != BackendWindowPhase::NestedFold) {
        continue;
      }
      std::uint32_t &advance = deferred_inner_advance[window->state];
      std::uint32_t &bound = deferred_inner_bound[window->state];
      if ((advance != unset && advance != window->inner_advance) ||
          (bound != unset && bound != window->inner_bound)) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      advance = window->inner_advance;
      bound = window->inner_bound;
    }
    native_publications.reserve(publications.size());
    native_windows.reserve(entries.size());
    MetalResidentState &resident = MetalResidents(*context.adapter);
    std::lock_guard resident_lock{resident.mutex};
    for (const BackendPublish &publication : publications) {
      const bool window = publication.kind == BackendPublishKind::Window;
      const MetalResidentBufferResult target = ResolveMetalResidentBuffer(
          resident, publication.target, publication.target_handle,
          "accel_metal_resident_id_unavailable", true);
      std::array<MetalResidentBufferResult, 3u> sources{};
      MetalResidentBufferResult count{};
      if (window) {
        count = ResolveMetalResidentBuffer(
            resident, publication.count.source, publication.count.handle,
            "accel_metal_resident_id_unavailable", true);
      }
      if (!target.check.ok || target.device_buffer == nullptr ||
          publication.state == std::numeric_limits<std::uint32_t>::max() ||
          (!window && publication.final >= publication.sources.size()) ||
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
            sources[bank].device_buffer != target.device_buffer &&
            source.source.count ==
                (window ? publication.tile : publication.target.count) &&
            source.source.element_bytes == publication.target.element_bytes &&
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
      state_count = std::max(state_count, publication.state + 1u);
      native_publications.push_back(MetalPublish{
          .sources = {sources[0].device_buffer, sources[1].device_buffer,
                      sources[2].device_buffer},
          .target = target.device_buffer,
          .count = window ? count.device_buffer : nullptr,
          .params =
              MetalPublishParams{
                  .count = publication.target.count,
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
                      publication.target.offset_bytes / sizeof(std::uint32_t),
                  .target_stride_words =
                      publication.target.stride_bytes / sizeof(std::uint32_t),
                  .element_words = static_cast<std::uint32_t>(
                      publication.target.element_bytes / sizeof(std::uint32_t)),
                  .declared_step_count = status.declared_step_count,
                  .state = publication.state,
                  .final = publication.final,
                  .maximum = publication.maximum,
                  .tile = publication.tile,
                  .kind = static_cast<std::uint32_t>(publication.kind),
                  .count_offset_words =
                      window ? publication.count.source.offset_bytes /
                                   sizeof(std::uint32_t)
                             : 0u,
              },
      });
    }
    for (std::size_t entry_index = 0u; entry_index < entries.size();
         ++entry_index) {
      const BackendWindow *const window =
          entries[entry_index].recurrence.window;
      if (window == nullptr) {
        continue;
      }
      const bool nested = window->nested();
      const std::uint32_t deferred =
          nested && window->state < deferred_inner_advance.size()
              ? deferred_inner_advance[window->state]
              : unset;
      const std::uint32_t deferred_bound =
          nested && window->state < deferred_inner_bound.size()
              ? deferred_inner_bound[window->state]
              : unset;
      const bool nested_shape_valid =
          !nested ||
          (window->outer_bound != 0u &&
           window->outer_iteration < window->outer_bound &&
           ((window->phase == BackendWindowPhase::NestedSeed &&
             window->route == 0u && window->inner_advance == 0u) ||
            (window->phase == BackendWindowPhase::NestedAction &&
             window->inner_bound != 0u &&
             window->inner_iteration < window->inner_bound &&
             window->route == 0u &&
             window->inner_advance ==
                 (entries[entry_index].transducer == NoTileTransducer ? 1u
                                                                      : 0u)) ||
            (window->phase == BackendWindowPhase::NestedFold &&
             window->route < 3u &&
             (window->inner_advance == 0u ||
              window->inner_advance == window->inner_bound))) &&
           deferred != unset && deferred_bound == window->inner_bound &&
           (deferred == 0u || deferred == deferred_bound));
      if (entries[entry_index].transducer != NoTileTransducer) {
        const TileTransducer &transducer =
            transducers[entries[entry_index].transducer];
        if (!nested || window->phase != BackendWindowPhase::NestedAction ||
            !transducer.recurrence.ready() ||
            transducer.template_first != entries[entry_index].template_index ||
            transducer.template_count != window->inner_bound) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
      }
      const MetalResidentBufferResult count = ResolveMetalResidentBuffer(
          resident, window->count.source, window->count.handle,
          "accel_metal_resident_id_unavailable", true);
      if (!count.check.ok || count.device_buffer == nullptr ||
          window->maximum == 0u || window->tile == 0u ||
          window->tile > window->maximum || window->bound == 0u ||
          window->iteration >= window->bound || !nested_shape_valid ||
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
                  .iteration = window->iteration,
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
                          ? deferred
                          : window->inner_advance,
              },
          .entry = static_cast<std::uint32_t>(entry_index),
      });
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
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
  }
  if (recurrence.ready()) {
    if (!transducers.empty()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (recurrence.first == nullptr || recurrence.windows == nullptr ||
        recurrence.window_count == 0u || recurrence.iterations < 2u ||
        recurrence.iterations > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck ready{};
    try {
      ready = PrepareMetalMap(
          *entries.front().run->pick, recurrence.plan, recurrence.artifact,
          recurrence.windows, recurrence.window_count, recurrence.bindings,
          recurrence.first->control, pipeline->recurrence,
          static_cast<std::uint32_t>(recurrence.iterations));
    } catch (const std::bad_alloc &) {
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
    const MapRecurrence &map = transducer.recurrence;
    if (!map.ready() || map.first == nullptr || map.windows == nullptr ||
        map.window_count == 0u || map.iterations < 2u ||
        map.iterations != transducer.template_count ||
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
      ready = PrepareMetalMap(*owner.run->pick, map.plan, map.artifact,
                              map.windows, map.window_count, map.bindings,
                              map.first->control, pipeline->transducers[index],
                              static_cast<std::uint32_t>(map.iterations));
    } catch (const std::bad_alloc &) {
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
