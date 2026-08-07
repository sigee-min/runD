#include "../build.hpp"

#include "../../../../kernel/backend/exception.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Describe() {
  // Direct aggregate admission already proved and resolved the complete
  // pipeline. Canonical status, telemetry, reset, and occurrence metadata
  // have no execution consumer on this path. Preserve only the public logical
  // status slices by inspecting one representative of each proved-identical
  // Seed/Action/Fold Program; do not retain raw arenas or occurrence records.
  if (aggregate_selected) {
    try {
      const NestedAggregate &aggregate = aggregates.front();
      const auto status_count = [&](const std::uint32_t template_index,
                                    std::uint32_t &out) {
        failure_context.template_route(template_index);
        if (template_index >= templates.size()) {
          return false;
        }
        const BackendBatchEntry &entry = templates[template_index];
        auto *const resources =
            entry.prepared == nullptr
                ? nullptr
                : static_cast<MetalKernelResources *>(entry.prepared->get());
        if (resources == nullptr) {
          return false;
        }
        out = 0u;
        return CountMetalDirectAggregateStatus(*resources, out);
      };
      std::uint32_t seed_status = 0u;
      std::uint32_t action_status = 0u;
      std::uint32_t fold_status = 0u;
      if (!status_count(aggregate.shape.seed_first(), seed_status) ||
          !status_count(aggregate.shape.action_first(), action_status) ||
          !status_count(aggregate.shape.fold_first(), fold_status)) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
      for (std::uint32_t index = 0u; index < status.active_step_count;
           ++index) {
        const std::uint32_t count =
            index < aggregate.shape.action_first()
                ? seed_status
                : (index < aggregate.shape.fold_first() ? action_status
                                                        : fold_status);
        if (!SetPreparedProgramStatusSlice(status, index, count)) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
      }
      status_entry_count = status.status_entry_count;
      return rund::AccelCheck{true, "ok"};
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  std::size_t template_step_capacity = 0u;
  for (const BackendBatchEntry &entry : templates) {
    const auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<const MetalKernelResources *>(entry.prepared->get());
    if (resources == nullptr || resources->size() == 0u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (resources->size() >
        std::numeric_limits<std::size_t>::max() - template_step_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    template_step_capacity += resources->size();
  }
  if (!template_registry.limit.ok ||
      template_step_capacity >
          template_registry.limit.backend_step_description_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const PreparedKernelPipelineReservation &limit = template_registry.limit;
  if (limit.backend_step_description_count >
          std::numeric_limits<std::size_t>::max() ||
      limit.backend_status_source_count >
          std::numeric_limits<std::size_t>::max() ||
      limit.backend_status_entry_count >
          std::numeric_limits<std::uint32_t>::max() ||
      limit.backend_telemetry_count > std::numeric_limits<std::size_t>::max()) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const std::size_t status_source_capacity =
      static_cast<std::size_t>(limit.backend_status_source_count);
  const std::uint32_t status_entry_capacity =
      static_cast<std::uint32_t>(limit.backend_status_entry_count);
  const std::size_t telemetry_capacity =
      static_cast<std::size_t>(limit.backend_telemetry_count);
  try {
    status_bindings.reserve(status_source_capacity);
    status_sources.reserve(status_source_capacity);
    status_resets.reserve(status_source_capacity);
    telemetry_steps.reserve(template_step_capacity);
    pipeline->telemetry.reserve(telemetry_capacity);
    if (status_bindings.capacity() != status_source_capacity ||
        status_sources.capacity() != status_source_capacity ||
        status_resets.capacity() != status_source_capacity ||
        telemetry_steps.capacity() != template_step_capacity ||
        pipeline->telemetry.capacity() != telemetry_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Describe immutable status and telemetry ownership exactly once per compact
  // route template. Physical nested occurrences reference these slices during
  // capture instead of manufacturing K*N status arenas or retained owners.
  for (std::size_t template_index = 0u; template_index < templates.size();
       ++template_index) {
    const BackendBatchEntry &entry = templates[template_index];
    failure_context.template_route(static_cast<std::uint32_t>(template_index));
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
    const std::uint32_t declared_step = status.declared_steps[template_index];
    if (entry.template_index != template_index ||
        declared_step >= status.declared_step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::size_t telemetry_begin = telemetry_steps.size();
    for (std::size_t step_index = 0u; step_index < resources->size();
         ++step_index) {
      failure_context.template_node_route(entry, step_index);
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
        if (pipeline->telemetry.size() >= telemetry_capacity) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        try {
          pipeline->telemetry.push_back(MetalPipelineTelemetryRecord{
              .source = source,
              .owner = step->resource,
          });
        } catch (...) {
          backend_exception::RethrowUnlessCapacityException();
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        slice.count = 1u;
      }
      if (telemetry_steps.size() >= template_step_capacity) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      try {
        telemetry_steps.push_back(slice);
      } catch (...) {
        backend_exception::RethrowUnlessCapacityException();
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    failure_context.template_route(static_cast<std::uint32_t>(template_index));
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
    bool status_capacity_failed = false;
    try {
      if (!CollectMetalStatus(*resources, declared_step, status_bindings,
                              status_sources, raw_status_count,
                              status_entry_count, status_source_capacity,
                              status_entry_capacity, status_capacity_failed)) {
        return rund::AccelCheck{false,
                                status_capacity_failed
                                    ? "compute_pipeline_capacity"
                                    : "accel_kernel_primitive_unsupported"};
      }
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
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
    failure_context.occurrence_route(entry);
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
    const TileTransducer *const transducer =
        entry.transducer == NoTileTransducer ? nullptr
                                             : &transducers[entry.transducer];
    const std::uint64_t physical_dispatches =
        transducer == nullptr ? resources->dispatch_count
                              : transducer->recurrence.window_count;
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
      if (physical_dispatches > std::numeric_limits<std::uint64_t>::max() -
                                    pipeline->dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->dispatch_count += physical_dispatches;
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
              : (transducer == nullptr ? entry.run->final_dispatch_count
                                       : physical_dispatches);
      const std::uint64_t physical =
          recurrence.ready() ? (recurrence_owner ? recurrence.window_count : 0u)
                             : physical_dispatches;
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
  if (profile_steps) {
    if (transducers.size() > PreparedPipelineStepCapacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    // Transducer identity is already bounded by the frozen compact route
    // table. Profiling only needs one counter per identity, so a heap mirror
    // would add a cold owner without adding information.
    std::array<std::uint64_t, PreparedPipelineStepCapacity> occurrence_counts{};
    for (const BackendBatchEntry &entry : entries) {
      if (entry.transducer != NoTileTransducer) {
        occurrence_counts[entry.transducer] =
            ::rund::detail::counter::SaturatingAdd(
                occurrence_counts[entry.transducer], 1u);
      }
    }
    for (std::size_t index = 0u; index < transducers.size(); ++index) {
      const TileTransducer &transducer = transducers[index];
      for (std::uint32_t offset = 1u; offset < transducer.template_count;
           ++offset) {
        const std::size_t template_index = transducer.template_first + offset;
        failure_context.template_route(
            static_cast<std::uint32_t>(template_index));
        const BackendBatchEntry &entry = templates[template_index];
        const std::uint32_t declared = status.declared_steps[template_index];
        if (entry.run == nullptr || declared >= status.declared_step_count) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        const std::uint64_t original =
            entry.run->original_dispatch_count != 0u &&
                    occurrence_counts[index] >
                        std::numeric_limits<std::uint64_t>::max() /
                            entry.run->original_dispatch_count
                ? std::numeric_limits<std::uint64_t>::max()
                : occurrence_counts[index] * entry.run->original_dispatch_count;
        PreparedPipelineStepEvidence &row = pipeline->step_evidence[declared];
        row.original_dispatch_count = ::rund::detail::counter::SaturatingAdd(
            row.original_dispatch_count, original);
      }
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
    if (status_resets.size() >= status_source_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
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
  if (status_entry_count > status_entry_capacity ||
      status_sources.size() > status_source_capacity ||
      telemetry_steps.size() > template_step_capacity ||
      pipeline->telemetry.size() > telemetry_capacity) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  try {
    status_entries.resize(status_entry_count);
    if (status_entries.capacity() != status_entry_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::size_t status_index = 0u;
  for (std::size_t template_index = 0u; template_index < templates.size();
       ++template_index) {
    failure_context.template_route(static_cast<std::uint32_t>(template_index));
    const PreparedProgramStatusSlice bindings = binding_slices[template_index];
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
