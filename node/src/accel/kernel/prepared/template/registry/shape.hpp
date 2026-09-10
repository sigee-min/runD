#pragma once

// Included by prepared/template/registry.hpp inside rund::node::accel::detail.

// Exact pipeline-global dimensions that cannot be reconstructed from a list
// of Program routes. Compute freezes this beside the route descriptors.
struct PreparedKernelPipelineShape final {
  std::uint64_t publication_count{};
  std::uint64_t terminal_publication_count{};
  // Exact physical publication dispatch upper for one backend stream.
  // Terminal routes emit one canonicalization and one final publication;
  // window routes consume the nested Shape's projected outer bound.
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
  BackendWindowPhase phase{BackendWindowPhase::Ordinary};
  bool writes_each_iteration{};
  bool has_window{};
  bool has_terminal{};

  [[nodiscard]] constexpr bool
  operator==(const PreparedKernelRecurrenceIdentity &) const noexcept = default;
};

static_assert(sizeof(PreparedKernelRecurrenceIdentity) == 52u);
static_assert(alignof(PreparedKernelRecurrenceIdentity) ==
              alignof(std::uint32_t));
static_assert(offsetof(PreparedKernelRecurrenceIdentity, phase) == 48u);
static_assert(offsetof(PreparedKernelRecurrenceIdentity,
                       writes_each_iteration) == 49u);
static_assert(offsetof(PreparedKernelRecurrenceIdentity, has_window) == 50u);
static_assert(offsetof(PreparedKernelRecurrenceIdentity, has_terminal) == 51u);

inline void
SeedPreparedKernelRecurrenceFingerprint(std::uint64_t &hi,
                                        std::uint64_t &lo) noexcept {
  hi = 0x72756e442e726563ull;
  lo = 0x757272656e63652eull;
}

[[nodiscard]] inline bool MixPreparedKernelRecurrenceFingerprint(
    std::uint64_t &hi, std::uint64_t &lo,
    const PreparedKernelRecurrenceIdentity &identity) noexcept {
  std::uint32_t phase = 0u;
  if (!EncodeBackendWindowPhase(identity.phase, phase)) {
    return false;
  }
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  mix(hi, identity.logical_step);
  mix(lo, identity.iteration);
  mix(hi, identity.bound);
  mix(lo, static_cast<std::uint64_t>(identity.writes_each_iteration));
  mix(hi, static_cast<std::uint64_t>(identity.has_window));
  if (!identity.has_window) {
    return true;
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
  mix(hi, phase);
  mix(lo, static_cast<std::uint64_t>(identity.has_terminal));
  return true;
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
