#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::mul;

[[nodiscard]] PreparedKernelPipelineReservation plan_pipeline_structure_counts(
    const std::uint64_t authored_entry_count,
    const std::uint64_t occurrence_count, const std::uint64_t window_count,
    const std::uint64_t nested_group_count) noexcept {
  PreparedKernelPipelineReservation result{};
  result.ok = true;
  result.reason = "ok";
  result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  result.authored_entry_count = authored_entry_count;
  result.occurrence_count = occurrence_count;
  result.window_count = window_count;
  result.nested_group_count = nested_group_count;
  std::uint64_t bytes = 0u;
  std::uint64_t item = 0u;
  if (!mul(result.authored_entry_count, sizeof(BackendBatchEntry), item) ||
      !accumulate(bytes, item) ||
      !mul(result.occurrence_count, sizeof(BackendBatchEntry), item) ||
      !accumulate(bytes, item) ||
      !mul(result.occurrence_count, sizeof(std::uint8_t), item) ||
      !accumulate(bytes, item) ||
      !mul(result.window_count, sizeof(BackendWindow), item) ||
      !accumulate(bytes, item) ||
      !mul(result.nested_group_count, sizeof(TileTransducer), item) ||
      !accumulate(bytes, item) ||
      !mul(result.nested_group_count, sizeof(NestedAggregate), item) ||
      !accumulate(bytes, item) ||
      (result.nested_group_count != 0u &&
       (!mul(result.authored_entry_count, sizeof(std::uint32_t), item) ||
        !accumulate(bytes, item)))) {
    result.ok = false;
    result.reason = "compute_pipeline_capacity";
    return result;
  }
  result.host_bytes = bytes;
  return result;
}

[[nodiscard]] PreparedKernelPipelineReservation plan_pipeline_structure(
    const std::span<const BackendRecurrence> recurrences) noexcept {
  std::uint64_t occurrence_count = 0u;
  std::uint64_t window_count = 0u;
  std::uint64_t nested_group_count = 0u;
  for (std::size_t index = 0u; index < recurrences.size();) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      if (!accumulate(occurrence_count, 1u) ||
          (window != nullptr && !accumulate(window_count, 1u))) {
        return PreparedKernelPipelineReservation{
            .reason = "compute_pipeline_capacity"};
      }
      ++index;
      continue;
    }
    NestedTemplateGeometry geometry{};
    if (!ProveNestedTemplateGeometry(recurrences, index, geometry)) {
      return PreparedKernelPipelineReservation{.reason =
                                                   "accel_kernel_run_invalid"};
    }
    const std::uint64_t commands = geometry.shape().authored_occurrence_count();
    if (!accumulate(occurrence_count, commands) ||
        !accumulate(window_count, commands) ||
        !accumulate(nested_group_count, 1u)) {
      return PreparedKernelPipelineReservation{.reason =
                                                   "compute_pipeline_capacity"};
    }
    index = geometry.end();
  }
  return plan_pipeline_structure_counts(recurrences.size(), occurrence_count,
                                        window_count, nested_group_count);
}

} // namespace rund::node::accel::detail
