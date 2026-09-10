#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::EncodePrograms() {
  reset_command_count = captured.commands.size();
  import_count = 0u;
  if (recurrence.ready()) {
    return metal_pipeline_program_internal::EncodeRecurrence(*this);
  }

  bool scratch_seen = false;
  for (std::size_t entry_index = 0u; entry_index < entries.size();
       ++entry_index) {
    metal_pipeline_program_internal::ProgramEntry entry{};
    const rund::AccelCheck prepared =
        metal_pipeline_program_internal::PrepareEntry(*this, entry_index,
                                                      scratch_seen, entry);
    if (!prepared.ok) {
      return prepared;
    }
    const rund::AccelCheck preflight =
        metal_pipeline_program_internal::EncodeWindowControl(
            *this, entry_index, entry.resident_window, 0u);
    if (!preflight.ok) {
      return preflight;
    }
    const std::size_t program_command_begin = captured.commands.size();
    const rund::AccelCheck body = metal_pipeline_program_internal::EncodeBody(
        *this, entry, entry_index, program_command_begin);
    if (!body.ok) {
      return body;
    }
    failure_context.occurrence_route(entries[entry_index]);
    const rund::AccelCheck advanced =
        metal_pipeline_program_internal::EncodeWindowControl(
            *this, entry_index, entry.resident_window, 1u);
    if (!advanced.ok) {
      return advanced;
    }
    const std::size_t program_command_end = captured.commands.size();
    if (program_command_begin == program_command_end) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    if (profile_steps) {
      const MetalWork work = MeasureMetalWork(std::span<const MetalCommand>{
          captured.commands.data() + program_command_begin,
          program_command_end - program_command_begin});
      if (work.exact) {
        PreparedPipelineStepEvidence &row =
            pipeline
                ->step_evidence[status.declared_steps[entry.template_index]];
        if (work.workgroup_count > std::numeric_limits<std::uint64_t>::max() -
                                       row.workgroup_count ||
            work.work_item_count > std::numeric_limits<std::uint64_t>::max() -
                                       row.work_item_count) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        row.workgroup_count += work.workgroup_count;
        row.work_item_count += work.work_item_count;
      }
    }
    const rund::AccelCheck status =
        metal_pipeline_program_internal::EncodeStatus(*this, entry);
    if (!status.ok) {
      return status;
    }
    const rund::AccelCheck folded =
        metal_pipeline_program_internal::EncodeWindowControl(
            *this, entry_index, entry.resident_window, 2u);
    if (!folded.ok) {
      return folded;
    }
    const rund::AccelCheck publications =
        metal_pipeline_program_internal::EncodePublications(*this, entry);
    if (!publications.ok) {
      return publications;
    }
    captured.declared_step = std::numeric_limits<std::uint32_t>::max();
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
