#pragma once

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

// Backend-owned contribution for the Map recurrence groups proved by the
// common compiler. Group and template counts describe physical prepared
// owners, never authored iterations. A nested Action recurrence therefore
// contributes one route group independent of outer * inner, while immutable
// terminal/history variants may be shared by every equal group.
struct PreparedMapRecurrenceReservation final {
  std::uint64_t route_host_bytes{};
  std::uint64_t route_native_bytes{};
  std::uint64_t template_host_bytes{};
  std::uint64_t template_native_bytes{};
  std::uint64_t template_source_bytes{};
  std::uint64_t source_transient_bytes{};
  std::uint64_t group_count{};
  std::uint64_t history_group_count{};
  std::uint64_t template_count{};
  std::uint64_t terminal_template_group_capacity{};
  std::uint64_t history_template_group_capacity{};
  std::uint64_t route_step_count{};
  std::uint64_t template_step_count{};
  std::uint64_t descriptor_set_count{};
  std::uint64_t descriptor_count{};
  std::uint64_t route_native_allocation_count{};
  std::uint64_t template_native_allocation_count{};

  [[nodiscard]] constexpr bool
  operator==(const PreparedMapRecurrenceReservation &) const noexcept = default;
};

// Fixed-width, allocation-free description of the retained structures and
// explicitly requested native payload that cold Pipeline preparation will
// materialize. Opaque driver bookkeeping is deliberately represented by
// structural object counts rather than invented byte estimates.
struct PreparedKernelPipelineReservation final {
  std::uint64_t fingerprint_hi{};
  std::uint64_t fingerprint_lo{};
  std::uint64_t host_bytes{};
  std::uint64_t native_bytes{};
  std::uint64_t route_host_bytes{};
  std::uint64_t route_native_bytes{};
  std::uint64_t template_host_bytes{};
  std::uint64_t template_native_bytes{};
  std::uint64_t template_source_bytes{};
  // Largest temporary source allocation that can coexist with the retained
  // cache sources while one template is specialized. Registry publication is
  // serialized, so this is a high-water mark rather than an additive owner.
  std::uint64_t source_transient_bytes{};
  // Largest non-source host workspace used by one backend cold finalizer.
  // This is a one-shot high-water mark, never a retained warm-path owner.
  std::uint64_t host_transient_bytes{};
  std::uint64_t route_count{};
  std::uint64_t template_count{};
  std::uint64_t template_step_count{};
  // Backend-owned ceiling for semantic-deduplicated immutable template
  // steps. This is distinct from compact Pipeline route/status capacity.
  std::uint64_t template_step_capacity{
      std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t route_step_count{};
  std::uint64_t authored_entry_count{};
  std::uint64_t occurrence_count{};
  std::uint64_t window_count{};
  std::uint64_t nested_group_count{};
  // Backend pipeline structural high-water marks. These are counts and
  // explicit payload bytes only; opaque driver bookkeeping is never guessed.
  std::uint64_t backend_dispatch_count{};
  std::uint64_t backend_reset_dispatch_count{};
  std::uint64_t backend_window_dispatch_count{};
  std::uint64_t backend_indirect_dispatch_count{};
  std::uint64_t backend_window_state_count{};
  std::uint64_t backend_window_descriptor_state_count{};
  std::uint64_t backend_step_occurrence_count{};
  std::uint64_t backend_step_description_count{};
  // Immutable status/telemetry descriptions are retained once per compact
  // route template. Their encoded commands repeat for physical occurrences
  // and therefore have separate dimensions below.
  std::uint64_t backend_status_source_count{};
  std::uint64_t backend_status_entry_count{};
  std::uint64_t backend_telemetry_count{};
  std::uint64_t backend_status_command_count{};
  std::uint64_t backend_telemetry_command_count{};
  std::uint64_t backend_publication_count{};
  std::uint64_t backend_terminal_publication_count{};
  // Vulkan derives the conservative window-control upper as 2 * window_count:
  // at most one transition and one NestedSeed preflight per occurrence.
  std::uint64_t backend_window_control_command_count{};
  // Exact physical publication dispatch upper from the common shape helper.
  std::uint64_t backend_publication_command_count{};
  std::uint64_t backend_command_count{};
  // Device-calibrated ICB size-class reservation. `native_bytes` owns the
  // byte total; these fields expose its exact physical command-stream subset
  // and retained chunk cardinality for plan/materialization parity. They are
  // physical allocation gates and intentionally remain outside the semantic
  // Pipeline fingerprint.
  std::uint64_t backend_command_chunk_count{};
  std::uint64_t backend_command_native_bytes{};
  // Maximum non-guard binding prefix authored by any command producer in the
  // stream. This is an index-space high-water mark, not an additive row count.
  std::uint64_t backend_command_binding_slot_upper{};
  std::uint64_t backend_command_binding_count{};
  std::uint64_t backend_parameter_bytes{};
  std::uint64_t backend_profile_step_count{};
  std::uint64_t backend_profile_command_count{};
  std::uint64_t backend_query_count{};
  std::uint64_t backend_native_buffer_count{};
  std::uint64_t backend_native_object_count{};
  std::uint64_t descriptor_set_count{};
  std::uint64_t descriptor_count{};
  std::uint64_t native_allocation_count{};
  // Auditable subset of the generic byte/count totals above. This preserves
  // one accounting authority while making recurrence-private preparation
  // impossible to hide inside an opaque backend aggregate.
  PreparedMapRecurrenceReservation map_recurrence{};
  // Fixed-size common semantic proof upper. Any recurrence candidate reserves
  // one owner; only a whole-Pipeline recurrence materializes it.
  std::uint64_t service_free_direct_host_bytes{};
  std::uint64_t template_capacity{};
  std::uint64_t descriptor_set_capacity{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint64_t descriptor_capacity{std::numeric_limits<std::uint32_t>::max()};
  bool ok{};
  const char *reason{"compute_pipeline_capacity"};

  [[nodiscard]] constexpr bool operator==(
      const PreparedKernelPipelineReservation &) const noexcept = default;
};

// One backend's allocation-free contribution for one bound Program route.
// Route fields are charged for every occurrence binding. Template fields are
// charged only for the first structurally equal Program/layout template.
struct PreparedKernelRouteReservation final {
  std::uint64_t route_host_bytes{};
  std::uint64_t route_native_bytes{};
  std::uint64_t route_step_count{};
  std::uint64_t descriptor_set_count{};
  std::uint64_t descriptor_count{};
  std::uint64_t route_native_allocation_count{};
  std::uint64_t template_host_bytes{};
  std::uint64_t template_native_bytes{};
  std::uint64_t template_source_bytes{};
  std::uint64_t source_transient_bytes{};
  // Exact allocation-backed description workspace retained only while this
  // canonical route is folded into a backend Pipeline command stream.
  std::uint64_t host_transient_bytes{};
  std::uint64_t template_step_count{};
  std::uint64_t template_step_capacity{
      std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t template_native_allocation_count{};
  std::uint64_t template_capacity{};
  // Backend physical body/view dispatches for one route occurrence. Reset,
  // status, telemetry, window-control, and publication commands retain their
  // own fields and must not be folded into this count.
  std::uint64_t dispatch_count{};
  std::uint64_t capture_direct_dispatch_count{};
  std::uint64_t capture_indirect_dispatch_count{};
  // Maximum non-guard argument index plus one for this route's operation and
  // view encoders. Occurrence expansion repeats it; it never adds slot spaces.
  std::uint64_t capture_binding_slot_upper{};
  std::uint64_t reset_dispatch_count{};
  std::uint64_t status_entry_count{};
  std::uint64_t status_source_count{};
  std::uint64_t status_command_count{};
  std::uint64_t status_parameter_bytes{};
  std::uint64_t telemetry_source_count{};
};

} // namespace rund::node::accel::detail
