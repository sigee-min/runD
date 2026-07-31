#include "../build.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Describe() {
  try {
    const std::size_t binding_capacity =
        templates.size() * kMetalPipelineStatusBindingCapacity;
    status_bindings.reserve(binding_capacity);
    status_sources.reserve(binding_capacity);
    status_resets.reserve(binding_capacity);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Describe immutable status and telemetry ownership exactly once per compact
  // route template. Physical nested occurrences reference these slices during
  // capture instead of manufacturing K*N status arenas or retained owners.
  for (std::size_t template_index = 0u; template_index < templates.size();
       ++template_index) {
    const BackendBatchEntry &entry = templates[template_index];
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<MetalKernelResources *>(entry.prepared->get());
    MetalKernelContext current{};
    if (entry.run == nullptr || entry.run->pick == nullptr ||
        resources == nullptr || resources->size() == 0u ||
        !IsPipelinePrivatePreparation(resources->mode) ||
        !ValidateMetalKernelContext(*entry.run->pick, current).ok ||
        current.adapter != context.adapter) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint32_t declared_step =
        status.declared_steps[template_index];
    if (entry.template_index != template_index ||
        declared_step >= status.declared_step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::size_t telemetry_begin = telemetry_steps.size();
    for (std::size_t step_index = 0u; step_index < resources->size();
         ++step_index) {
      MetalKernelEntry *const step = resources->entry(step_index);
      if (step == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (pipeline->telemetry.size() >
          std::numeric_limits<std::uint32_t>::max()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      PreparedProgramStatusSlice slice{
          .first = static_cast<std::uint32_t>(pipeline->telemetry.size())};
      MetalPipelineTelemetrySource source{};
      if (step->ops.pipeline_telemetry != nullptr) {
        if (!step->ops.pipeline_telemetry(step->resource, source)) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
      }
      if (source.kind != MetalPipelineTelemetryKind::None) {
        try {
          pipeline->telemetry.push_back(MetalPipelineTelemetryRecord{
              .source = source,
              .owner = step->resource,
          });
        } catch (const std::bad_alloc &) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        slice.count = 1u;
      }
      try {
        telemetry_steps.push_back(slice);
      } catch (const std::bad_alloc &) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    const std::size_t telemetry_count =
        telemetry_steps.size() - telemetry_begin;
    if (telemetry_begin > std::numeric_limits<std::uint32_t>::max() ||
        telemetry_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    telemetry_ranges[template_index] = PreparedProgramStatusSlice{
        .first = static_cast<std::uint32_t>(telemetry_begin),
        .count = static_cast<std::uint32_t>(telemetry_count),
    };
    const std::size_t binding_begin = status_bindings.size();
    const std::uint32_t status_begin = status_entry_count;
    try {
      if (!CollectMetalStatus(*resources, declared_step, status_bindings,
                              status_sources, raw_status_count,
                              status_entry_count)) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::size_t binding_size = status_bindings.size() - binding_begin;
    const std::uint32_t status_size = status_entry_count - status_begin;
    if (binding_begin > std::numeric_limits<std::uint32_t>::max() ||
        binding_size > std::numeric_limits<std::uint32_t>::max() ||
        !SetPreparedProgramStatusSlice(
            status, static_cast<std::uint32_t>(template_index), status_size)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    binding_slices[template_index] = PreparedProgramStatusSlice{
        .first = static_cast<std::uint32_t>(binding_begin),
        .count = static_cast<std::uint32_t>(binding_size),
    };
  }
  // Dispatch/reset evidence follows physical command occurrences. Profile
  // rows aggregate repeated occurrences back into their declared compact step.
  for (std::size_t entry_index = 0u; entry_index < entries.size();
       ++entry_index) {
    const BackendBatchEntry &entry = entries[entry_index];
    if (entry.template_index >= templates.size() ||
        entry.occurrence_index != entry_index ||
        entry.run != templates[entry.template_index].run ||
        entry.prepared != templates[entry.template_index].prepared) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<MetalKernelResources *>(entry.prepared->get());
    if (resources == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint32_t declared_step =
        status.declared_steps[entry.template_index];
    if (declared_step >= status.declared_step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (recurrence.ready()) {
      if (entry_index == 0u) {
        pipeline->dispatch_count = recurrence.window_count;
      }
    } else {
      if (resources->dispatch_count >
          std::numeric_limits<std::uint64_t>::max() -
              pipeline->dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->dispatch_count += resources->dispatch_count;
      pipeline->reset_count = ::rund::detail::counter::SaturatingAdd(
          pipeline->reset_count, resources->reset_count);
      pipeline->reset_bytes = ::rund::detail::counter::SaturatingAdd(
          pipeline->reset_bytes, resources->reset_bytes);
    }
    if (profile_steps) {
      const bool recurrence_owner = recurrence.ready() && entry_index == 0u;
      PreparedPipelineStepEvidence &row =
          pipeline->step_evidence[declared_step];
      const std::uint64_t original = entry.run->original_dispatch_count;
      const std::uint64_t final =
          recurrence.ready()
              ? (recurrence_owner ? recurrence.window_count : 0u)
              : entry.run->final_dispatch_count;
      const std::uint64_t physical =
          recurrence.ready()
              ? (recurrence_owner ? recurrence.window_count : 0u)
              : resources->dispatch_count;
      if (original > std::numeric_limits<std::uint64_t>::max() -
                         row.original_dispatch_count ||
          final > std::numeric_limits<std::uint64_t>::max() -
                      row.final_dispatch_count ||
          physical > std::numeric_limits<std::uint64_t>::max() -
                         row.physical_dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      row.original_dispatch_count += original;
      row.final_dispatch_count += final;
      row.physical_dispatch_count += physical;
    }
  }
  // Private replacement ranges are packed first.  Only those words need an
  // initial value: public ranges are overwritten in full by their import
  // dispatch.  Reassigning raw offsets leaves canonical metadata order
  // untouched while turning the private prefix into one compact reset grid.
  private_raw_count = 0u;
  for (std::size_t index = 0u; index < status_bindings.size(); ++index) {
    MetalPipelineStatusBindingRecord &record = status_bindings[index];
    if (!record.binding.replace) {
      continue;
    }
    record.raw_offset = private_raw_count;
    private_raw_count += record.raw_count;
    status_resets.push_back(MetalPipelineResetMeta{
        .raw_offset = record.raw_offset,
        .reset = record.binding.reset,
    });
  }
  std::uint32_t public_raw_offset = private_raw_count;
  for (std::size_t index = 0u; index < status_bindings.size(); ++index) {
    MetalPipelineStatusBindingRecord &record = status_bindings[index];
    if (record.binding.replace) {
      continue;
    }
    record.raw_offset = public_raw_offset;
    public_raw_offset += record.raw_count;
  }
  for (std::size_t index = 0u; index < status_bindings.size(); ++index) {
    status_sources[index].raw_offset = status_bindings[index].raw_offset;
  }
  try {
    status_entries.resize(status_entry_count);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::size_t status_index = 0u;
  for (std::size_t template_index = 0u; template_index < templates.size();
       ++template_index) {
    const PreparedProgramStatusSlice bindings =
        binding_slices[template_index];
    const std::size_t binding_end =
        static_cast<std::size_t>(bindings.first) + bindings.count;
    if (binding_end > status_bindings.size()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    for (std::size_t source_index = bindings.first; source_index < binding_end;
         ++source_index) {
      const MetalPipelineStatusBindingRecord &record =
          status_bindings[source_index];
      if (source_index - bindings.first >
              std::numeric_limits<std::uint32_t>::max() ||
          status_index > status_entries.size() ||
          record.binding.observed_count >
              status_entries.size() - status_index) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const std::uint32_t source =
          static_cast<std::uint32_t>(source_index - bindings.first);
      for (std::uint32_t observed_index = 0u;
           observed_index < record.binding.observed_count; ++observed_index) {
        status_entries[status_index++] = MetalPipelineStatusEntryMeta{
            .source = source,
            .raw = record.raw_offset + record.binding.observed + observed_index,
        };
      }
    }
  }
  if (public_raw_offset != raw_status_count ||
      status.status_entry_count != status_entry_count ||
      status_sources.size() != status_bindings.size() ||
      status_index != status_entries.size() ||
      !ValidMetalReset(status_resets, private_raw_count)) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
