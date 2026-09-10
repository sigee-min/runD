#include "internal.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
DescribeMetalPacking(MetalPipelineBuild &build,
                     const MetalPipelineDescribeCapacity &capacity) {
  // Private replacement ranges are packed first. Only those words need an
  // initial value: public ranges are overwritten in full by their import
  // dispatch. Reassigning raw offsets leaves canonical metadata order
  // untouched while turning the private prefix into one compact reset grid.
  build.private_raw_count = 0u;
  for (std::size_t index = 0u; index < build.status_bindings.size(); ++index) {
    MetalPipelineStatusBindingRecord &record = build.status_bindings[index];
    if (!record.binding.replace) {
      continue;
    }
    record.raw_offset = build.private_raw_count;
    build.private_raw_count += record.raw_count;
    if (build.status_resets.size() >= capacity.status_source_capacity) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    build.status_resets.push_back(MetalPipelineResetMeta{
        .raw_offset = record.raw_offset,
        .reset = record.binding.reset,
    });
  }
  std::uint32_t public_raw_offset = build.private_raw_count;
  for (std::size_t index = 0u; index < build.status_bindings.size(); ++index) {
    MetalPipelineStatusBindingRecord &record = build.status_bindings[index];
    if (record.binding.replace) {
      continue;
    }
    record.raw_offset = public_raw_offset;
    public_raw_offset += record.raw_count;
  }
  for (std::size_t index = 0u; index < build.status_bindings.size(); ++index) {
    build.status_sources[index].raw_offset =
        build.status_bindings[index].raw_offset;
  }
  if (build.status_entry_count > capacity.status_entry_capacity ||
      build.status_sources.size() > capacity.status_source_capacity ||
      build.telemetry_steps.size() > capacity.template_step_capacity ||
      build.pipeline->telemetry.size() > capacity.telemetry_capacity) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  try {
    build.status_entries.resize(build.status_entry_count);
    if (build.status_entries.capacity() != build.status_entry_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::size_t status_index = 0u;
  for (std::size_t template_index = 0u; template_index < build.templates.size();
       ++template_index) {
    build.failure_context.template_route(
        static_cast<std::uint32_t>(template_index));
    const PreparedProgramStatusSlice bindings =
        build.binding_slices[template_index];
    const std::size_t binding_end =
        static_cast<std::size_t>(bindings.first) + bindings.count;
    if (binding_end > build.status_bindings.size()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    for (std::size_t source_index = bindings.first; source_index < binding_end;
         ++source_index) {
      const MetalPipelineStatusBindingRecord &record =
          build.status_bindings[source_index];
      if (source_index - bindings.first >
              std::numeric_limits<std::uint32_t>::max() ||
          status_index > build.status_entries.size() ||
          record.binding.observed_count >
              build.status_entries.size() - status_index) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const std::uint32_t source =
          static_cast<std::uint32_t>(source_index - bindings.first);
      for (std::uint32_t observed_index = 0u;
           observed_index < record.binding.observed_count; ++observed_index) {
        build.status_entries[status_index++] = MetalPipelineStatusEntryMeta{
            .source = source,
            .raw = record.raw_offset + record.binding.observed + observed_index,
        };
      }
    }
  }
  if (public_raw_offset != build.raw_status_count ||
      build.status.status_entry_count != build.status_entry_count ||
      build.status_sources.size() != build.status_bindings.size() ||
      status_index != build.status_entries.size() ||
      !ValidMetalReset(build.status_resets, build.private_raw_count)) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal
