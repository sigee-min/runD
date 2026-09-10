#include "internal.hpp"

#include "../../../../buffer/resident/find.hpp"
#include "../../../../resident.hpp"
#include "../../../../resident/access.hpp"
#include "../../../../runtime/map/api.hpp"
#include "../../../../runtime/map/resources.hpp"

#include "../../../../../kernel/backend/exception.hpp"
#include "../../../../../kernel/recurrence/plan.hpp"
#include "../../../../../kernel/step/map/stride.hpp"

#include "../../../../pipeline/guard.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail::metal_pipeline_admit_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
PrepareMetalPipelineResources(MetalPipelineBuild &build,
                              MetalSpatialWindowProof &spatial_window,
                              std::uint32_t &state_count) {
  build.recurrence = BuildMapRecurrence(build.entries, build.barriers);
  if (build.recurrence.invalid()) {
    return rund::AccelCheck{false, build.recurrence.reason};
  }
  state_count = 0u;
  try {
    spatial_window = ProveMetalSpatialWindow(
        build.entries, build.recurrence, build.transducers, build.aggregates,
        build.publications, build.status.declared_step_count);
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
    for (const BackendBatchEntry &entry : build.entries) {
      build.failure_context.occurrence_route(entry);
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
    build.native_windows.reserve(planned_native_window_count);
    if (build.native_windows.capacity() != planned_native_window_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    MetalResidentState &resident = MetalResidents(*build.context.adapter);
    std::lock_guard resident_lock{resident.mutex};
    build.failure_context.clear_route();
    for (const BackendPublish &publication : build.publications) {
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
      if (build.native_publication_count == build.native_publications.size()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      build.native_publications[build.native_publication_count++] =
          MetalPublish{
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
                      .declared_step_count = build.status.declared_step_count,
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
    for (std::size_t entry_index = 0u; entry_index < build.entries.size();
         ++entry_index) {
      build.failure_context.occurrence_route(build.entries[entry_index]);
      const BackendWindow *const window =
          build.entries[entry_index].recurrence.window;
      if (window == nullptr) {
        continue;
      }
      const bool nested = window->nested();
      if (window->state == unset || window->state >= deferred_inner.size()) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      const DeferredInnerState &deferred = deferred_inner[window->state];
      const bool occurrence_valid = window->valid_occurrence(
          build.entries[entry_index].transduced_action());
      const bool deferred_valid =
          !nested ||
          (deferred.used && deferred.bound == window->inner_bound &&
           (deferred.advance == 0u || deferred.advance == deferred.bound));
      if (build.entries[entry_index].transducer != NoTileTransducer) {
        const TileTransducer &transducer =
            build.transducers[build.entries[entry_index].transducer];
        if (!occurrence_valid || !transducer.recurrence.ready() ||
            transducer.template_first !=
                build.entries[entry_index].template_index ||
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
      const std::uint32_t template_index =
          build.entries[entry_index].template_index;
      if (template_index >= build.status.active_step_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      std::uint32_t phase = 0u;
      if (!EncodeBackendWindowPhase(window->phase, phase)) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      build.native_windows.push_back(MetalWindow{
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
                  .phase = phase,
                  .declared_step = build.status.declared_steps[template_index],
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
    if (build.native_windows.size() != planned_native_window_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  build.failure_context.clear_route();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_admit_internal
