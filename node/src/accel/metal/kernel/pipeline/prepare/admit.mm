#include "../build.hpp"

#include "../../../buffer/resident/find.hpp"
#include "../../../resident.hpp"
#include "../../../resident/access.hpp"
#include "../../../runtime/map/api.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Admit() {
  if (templates.empty() || entries.empty() ||
      entries.size() != barriers.size() ||
      templates.size() != status.active_step_count ||
      entries.size() != status.command_count ||
      entries.front().run == nullptr || entries.front().run->pick == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    if (entries[index].occurrence_index != index ||
        entries[index].template_index >= templates.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  recurrence = BuildMapRecurrence(entries, barriers);
  if (recurrence.invalid()) {
    return rund::AccelCheck{false, recurrence.reason};
  }
  const rund::AccelCheck valid =
      ValidateMetalKernelContext(*entries.front().run->pick, context);
  if (!valid.ok || context.adapter == nullptr) {
    return valid;
  }
  std::uint32_t state_count = 0u;
  try {
    native_publications.reserve(publications.size());
    native_windows.reserve(entries.size());
    MetalResidentState &resident = MetalResidents(*context.adapter);
    std::lock_guard resident_lock{resident.mutex};
    for (const BackendPublish &publication : publications) {
      const MetalResidentBufferResult target = ResolveMetalResidentBuffer(
          resident, publication.target, publication.target_handle,
          "accel_metal_resident_id_unavailable", true);
      std::array<MetalResidentBufferResult, 3u> sources{};
      if (!target.check.ok || target.device_buffer == nullptr ||
          publication.state == std::numeric_limits<std::uint32_t>::max() ||
          publication.final >= publication.sources.size()) {
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
            source.source.count == publication.target.count &&
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
      const bool nested_shape_valid =
          !nested ||
          (window->outer_bound != 0u &&
           window->outer_iteration < window->outer_bound &&
           window->inner_bound != 0u &&
           ((window->phase == BackendWindowPhase::NestedSeed &&
             window->route == 0u) ||
            (window->phase == BackendWindowPhase::NestedAction &&
             window->inner_iteration < window->inner_bound &&
             window->route == 0u) ||
            (window->phase == BackendWindowPhase::NestedFold &&
             window->route < 3u)));
      const MetalResidentBufferResult count = ResolveMetalResidentBuffer(
          resident, window->count.source, window->count.handle,
          "accel_metal_resident_id_unavailable", true);
      if (!count.check.ok || count.device_buffer == nullptr ||
          window->maximum == 0u || window->tile == 0u ||
          window->tile > window->maximum || window->bound == 0u ||
          window->iteration >= window->bound ||
          !nested_shape_valid ||
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
                  .range_count = 0u,
                  .phase = static_cast<std::uint32_t>(window->phase),
                  .declared_step = status.declared_steps[template_index],
                  .overflow_reason = static_cast<std::uint32_t>(
                      rund::compute::Reason::BoundedCountInvalid),
                  .inner_bound = window->inner_bound,
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
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
