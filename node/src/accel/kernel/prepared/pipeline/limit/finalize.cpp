#include "internal.hpp"

#include "../../../recurrence.hpp"
#include "../../evidence.hpp"
#include "../backend.hpp"
#include "../registry.hpp"
#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::node::accel::detail::prepared_pipeline_limit {
namespace {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

} // namespace

bool finalize(const rund::AccelContext &context,
              const PreparedKernelPipelineShape shape, State &state) noexcept {
  PreparedKernelPipelineReservation &result = state.result;
  const PreparedKernelPipelineReservation structure =
      plan_pipeline_structure_counts(
          result.authored_entry_count, result.occurrence_count,
          result.window_count, result.nested_group_count);
  PreparedKernelPipelineReservation backend_structure{};
  PreparedKernelPipelineReservation per_stream_backend{};
  const bool divisible_stream_shape =
      state.stream_count != 0u &&
      structure.occurrence_count % state.stream_count == 0u &&
      structure.window_count % state.stream_count == 0u &&
      structure.nested_group_count % state.stream_count == 0u;
  const std::uint64_t per_stream_occurrences =
      divisible_stream_shape
          ? structure.occurrence_count / state.stream_count
          : 0u;
  if (divisible_stream_shape) {
    per_stream_backend.occurrence_count = per_stream_occurrences;
    per_stream_backend.window_count =
        structure.window_count / state.stream_count;
    per_stream_backend.nested_group_count =
        structure.nested_group_count / state.stream_count;
  }
  const bool backend_projection_matches =
      structure.ok && state.ops != nullptr && divisible_stream_shape &&
      state.backend_projection.occurrence_count == per_stream_occurrences;
  const rund::AccelCheck backend_planned =
      backend_projection_matches
          ? finalize_pipeline_backend_structure(
                context, *state.ops, state.backend_projection,
                shape.publication_count, shape.terminal_publication_count,
                shape.backend_publication_command_count,
                shape.window_state_count, shape.window_descriptor_state_count,
                shape.profile_steps ? shape.declared_step_count : 0u,
                shape.profile_steps ? per_stream_occurrences : 0u,
                per_stream_backend)
          : rund::AccelCheck{false, "compute_pipeline_capacity"};
  if (backend_planned.ok) {
    // Common expanded structure owns these dimensions. The backend only sizes
    // its private control owners from the per-stream projection.
    per_stream_backend.occurrence_count = 0u;
    per_stream_backend.window_count = 0u;
    per_stream_backend.nested_group_count = 0u;
    for (std::uint64_t stream = 0u; stream < state.stream_count; ++stream) {
      if (!accumulate_reservation(backend_structure, per_stream_backend)) {
        result.reason = "compute_pipeline_capacity";
        return false;
      }
    }
  }
  std::uint64_t pipeline_headers = 0u;
  std::uint64_t state_pointers = 0u;
  std::uint64_t registry_bytes = 0u;
  result.service_free_direct_host_bytes =
      result.map_recurrence.group_count == 0u ? 0u
                                              : sizeof(ServiceFreeDirectProof);
  if (!structure.ok || !backend_planned.ok || state.ops == nullptr ||
      result.route_count == 0u || result.template_count == 0u ||
      result.template_count > result.template_capacity ||
      !mul(state.stream_count, sizeof(prepared::PipelineState),
           pipeline_headers) ||
      !mul(state.entry_count, sizeof(std::shared_ptr<prepared::RunState>),
           state_pointers) ||
      !PreparedKernelTemplateRegistryBytes(result.template_count,
                                           registry_bytes) ||
      !add(pipeline_headers, state_pointers, result.host_bytes) ||
      !add(result.route_native_bytes, result.template_native_bytes,
           result.native_bytes) ||
      !accumulate(result.host_bytes, registry_bytes) ||
      !accumulate(result.host_bytes, structure.host_bytes) ||
      !accumulate_reservation(result, backend_structure) ||
      !accumulate(result.host_bytes, result.route_host_bytes) ||
      !accumulate(result.host_bytes, result.template_host_bytes) ||
      !accumulate(result.host_bytes, result.service_free_direct_host_bytes) ||
      !accumulate(result.host_bytes, result.source_transient_bytes) ||
      !accumulate(result.host_bytes, result.host_transient_bytes) ||
      result.descriptor_set_count > result.descriptor_set_capacity ||
      result.descriptor_count > result.descriptor_capacity ||
      result.host_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  result.ok = true;
  result.reason = "ok";
  return true;
}

} // namespace rund::node::accel::detail::prepared_pipeline_limit
