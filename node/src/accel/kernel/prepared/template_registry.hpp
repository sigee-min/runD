#pragma once

#include "../memory.hpp"
#include "../scratch.hpp"
#include "../view.hpp"

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/kernel/value.hpp>

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

struct BackendRun;
struct KernelExecutionStep;
struct KernelExecution;
struct BackendOps;

// Exact pipeline-global dimensions that cannot be reconstructed from a list
// of Program routes. Compute freezes this beside the route descriptors.
struct PreparedKernelPipelineShape final {
  std::uint64_t publication_count{};
  std::uint64_t terminal_publication_count{};
  // Exact physical publication dispatch upper for one backend stream.
  // Terminal routes emit one canonicalization and one final publication;
  // window routes publish once per maximum/tile window.
  std::uint64_t backend_publication_command_count{};
  // Per-stream Vulkan window identities. `window_state_count` is the exact
  // max active state id plus one (native state buffer extent), while
  // `window_descriptor_state_count` is the number of distinct active states
  // that each own one deduplicated window descriptor.
  std::uint64_t window_state_count{};
  std::uint64_t window_descriptor_state_count{};
  std::uint64_t publication_fingerprint_hi{};
  std::uint64_t publication_fingerprint_lo{};
  std::uint64_t declared_step_count{};
  std::uint32_t route_copies{1u};
  bool profile_steps{};
};

// Sole arithmetic authority for converting one semantic publication into its
// physical backend command contribution. Terminal publications have no window
// extent and emit canonicalize + final. Window publications emit once for each
// ceil(maximum / tile) outer window.
[[nodiscard]] inline bool PreparedKernelPublicationCommandContribution(
    const bool window, const std::uint64_t maximum, const std::uint64_t tile,
    std::uint64_t &commands) noexcept {
  commands = 0u;
  if (!window) {
    if (maximum != 0u || tile != 0u) {
      return false;
    }
    commands = 2u;
    return true;
  }
  if (maximum == 0u || tile == 0u || tile > maximum) {
    return false;
  }
  commands = maximum / tile;
  return maximum % tile == 0u ||
         rund::kernel::checked::add(commands, 1u, commands);
}

// Stable publication identity shared by the public Compute plan and private
// backend materialization. Resource ordinals come from Compute's canonical
// resource admission order; pointers and native handles are intentionally not
// part of this descriptor. Every ResidentBufferRef field that changes the
// bytes addressed or the required access is represented explicitly.
struct PreparedKernelPublicationViewIdentity final {
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t count{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint32_t resource_ordinal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t usage{};
};

struct PreparedKernelPublicationIdentity final {
  PreparedKernelPublicationViewIdentity sources[3]{};
  PreparedKernelPublicationViewIdentity count{};
  PreparedKernelPublicationViewIdentity target{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t final{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint8_t kind{};
};

// Exact, pointer-free shape of one Program graph binding before private Jobs
// exist. Public Pipeline planning freezes these in Program binding order.
// Backend template identity projects only Map data bindings through the
// admitted KernelExecution indices, matching private MapBindingFor exactly.
struct PreparedKernelProgramBindingIdentity final {
  std::uint64_t offset_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t count{};
  std::uint32_t usage{};
};

inline void
SeedPreparedKernelPublicationFingerprint(std::uint64_t &hi,
                                         std::uint64_t &lo) noexcept {
  hi = 0x72756e442e707562ull;
  lo = 0x6c69636174696f6eull;
}

inline void MixPreparedKernelPublicationFingerprint(
    std::uint64_t &hi, std::uint64_t &lo,
    const PreparedKernelPublicationIdentity &identity) noexcept {
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  const auto view = [&](const PreparedKernelPublicationViewIdentity &value) {
    mix(hi, value.resource_ordinal);
    mix(lo, value.backing_bytes);
    mix(hi, value.offset_bytes);
    mix(lo, value.count);
    mix(hi, value.stride_bytes);
    mix(lo, value.element_bytes);
    mix(hi, value.usage);
  };
  for (const PreparedKernelPublicationViewIdentity &source : identity.sources) {
    view(source);
  }
  view(identity.count);
  view(identity.target);
  mix(lo, identity.state);
  mix(hi, identity.final);
  mix(lo, identity.maximum);
  mix(hi, identity.tile);
  mix(lo, identity.kind);
}

// Allocation-free semantic projection shared by Compute's public plan and the
// backend materializer. Pointer/handle identity is intentionally absent: the
// exact resident view/scratch tuples are fingerprinted separately, while this
// descriptor freezes authored recurrence meaning and coordinates.
struct PreparedKernelRecurrenceIdentity final {
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t bound{1u};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t expected{};
  std::uint32_t outer_iteration{};
  std::uint32_t outer_bound{};
  std::uint32_t inner_iteration{};
  std::uint32_t inner_bound{};
  std::uint32_t route{};
  std::uint32_t state{};
  std::uint8_t phase{};
  bool writes_each_iteration{};
  bool has_window{};
  bool has_terminal{};
};

inline void
SeedPreparedKernelRecurrenceFingerprint(std::uint64_t &hi,
                                        std::uint64_t &lo) noexcept {
  hi = 0x72756e442e726563ull;
  lo = 0x757272656e63652eull;
}

inline void MixPreparedKernelRecurrenceFingerprint(
    std::uint64_t &hi, std::uint64_t &lo,
    const PreparedKernelRecurrenceIdentity &identity) noexcept {
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  mix(hi, identity.logical_step);
  mix(lo, identity.iteration);
  mix(hi, identity.bound);
  mix(lo, static_cast<std::uint64_t>(identity.writes_each_iteration));
  mix(hi, static_cast<std::uint64_t>(identity.has_window));
  if (!identity.has_window) {
    return;
  }
  mix(lo, identity.maximum);
  mix(hi, identity.tile);
  mix(lo, identity.expected);
  mix(hi, identity.outer_iteration);
  mix(lo, identity.outer_bound);
  mix(hi, identity.inner_iteration);
  mix(lo, identity.inner_bound);
  mix(hi, identity.route);
  mix(lo, identity.state);
  mix(hi, identity.phase);
  mix(lo, static_cast<std::uint64_t>(identity.has_terminal));
}

// Public Pipeline planning has Program identity and route shape before private
// Jobs, bound Buffer handles, or backend objects exist. One entry describes a
// canonical private-Job owner. `route_copies` is two only for transactional
// primary/alternate streams. The backend registry, not this public route list,
// decides collision-safe structural template equivalence at materialization.
struct PreparedKernelProgramRoute final {
  const rund::AccelKernel *kernel{};
  std::uint64_t tile_count{};
  const KernelViewLayout *views{};
  const KernelScratchLayout *scratch{};
  std::span<const PreparedKernelProgramBindingIdentity> program_bindings{};
  // Number of compact backend table entries that borrow this one canonical
  // route owner (normally one; recurrence parity reuse may be greater).
  std::uint64_t entry_count{1u};
  // Allocation-free upper bounds for the canonical (unfused) expansion that
  // references this owner in one stream. Backends may fuse occurrences but
  // may never materialize more than these public-plan counts.
  std::uint64_t occurrence_count{1u};
  std::uint64_t window_count{};
  // Charged once by the designated owner of each nested Seed/Action/Fold
  // group. It bounds both the optional transducer and aggregate proof tables.
  std::uint64_t nested_group_count{};
  // Candidate Map recurrence proofs assigned to this canonical route owner.
  // A top-level recurrence contributes at most one group for the complete
  // Pipeline; each nested window contributes at most one Action group,
  // independent of its outer/inner occurrence product. History is a strict
  // subset and selects a different transformed-source variant.
  std::uint64_t map_recurrence_group_count{};
  std::uint64_t map_recurrence_history_group_count{};
  std::uint64_t recurrence_fingerprint_hi{};
  std::uint64_t recurrence_fingerprint_lo{};
  std::uint32_t route_copies{1u};
};

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

// Compute owns this value and may share one instance across primary and
// transactional-alternate streams. Planning writes only the fixed reservation;
// the opaque owner is allocated later, after the caller accepts that plan.
struct PreparedKernelTemplateRegistry final {
  std::shared_ptr<void> owner{};
  // Frozen public-plan authority. Runtime preparation may consume less but
  // cannot enlarge any retained byte or structural count after this point.
  PreparedKernelPipelineReservation limit{};
  PreparedKernelPipelineReservation reservation{};
};

using PreparedKernelTemplateMatch = bool (*)(const void *prepared,
                                             const void *probe) noexcept;

// Backend template lookup is keyed first by the immutable admitted step
// authority, then by a cheap variant partition. Match is the collision-safe
// semantic equality check and is mandatory.
[[nodiscard]] std::shared_ptr<void>
FindPreparedKernelTemplate(const PreparedKernelTemplateRegistry &registry,
                           const KernelExecutionStep *authority,
                           std::uint64_t variant_hi, std::uint64_t variant_lo,
                           PreparedKernelTemplateMatch match,
                           const void *probe) noexcept;

// Publishes one fully prepared immutable template. If another equal template
// already won publication, `prepared` is replaced with that sole owner.
[[nodiscard]] rund::AccelCheck PublishPreparedKernelTemplate(
    PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *authority, std::uint64_t variant_hi,
    std::uint64_t variant_lo, const BackendOps &ops,
    PreparedKernelTemplateMatch match, const void *probe,
    std::shared_ptr<void> &prepared) noexcept;

// Materialization initializes/validates the opaque registry only after the
// allocation-free reservation was accepted.
[[nodiscard]] rund::AccelCheck BindPreparedKernelTemplateRegistry(
    const rund::AccelApi api, std::uint64_t context_id,
    PreparedKernelTemplateRegistry &registry) noexcept;

} // namespace rund::node::accel::detail
