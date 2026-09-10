#pragma once

// Included by backend/run.hpp inside rund::node::accel::detail.

struct ResidentState final {
  std::uint32_t current{};
  std::uint32_t stopped{};
};

static_assert(std::is_standard_layout_v<ResidentState>);
static_assert(sizeof(ResidentState) == 2u * sizeof(std::uint32_t));
static_assert(alignof(ResidentState) == alignof(std::uint32_t));
static_assert(offsetof(ResidentState, current) == 0u);
static_assert(offsetof(ResidentState, stopped) == 4u);

struct BackendWindow;

// Authored recurrence identity is carried from Pipeline planning so a backend
// never guesses that an ordinary ping-pong chain is disposable. A bound of one
// is the canonical non-recurrence value.
struct BackendRecurrence final {
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t bound{1u};
  const BackendWindow *window{};
  // An externally retained occurrence is semantically observable after the
  // enclosing submission. A terminal-only recurrence transform must not
  // replace the authored per-iteration stores with one final store.
  bool writes_each_iteration{};
};

struct BackendRead final {
  rund::kernel::ResidentBufferRef source{};
  std::shared_ptr<void> handle{};
};

// One compact route descriptor for a bounded resident recurrence. Ordinary
// routes already name one physical occurrence. The fallback nested path copies
// a template and writes that occurrence's outer/inner coordinates into the
// same fields before backend admission. A separately proved compact aggregate
// consumes template identity without treating its placeholder as an occurrence
// coordinate. The three terminal routes are the seed, first bank, and second
// bank. One fixed-width ResidentState selector owns the logical transition; an
// inactive occurrence leaves it unchanged and never copies payload between
// banks.
struct BackendWindow final {
  BackendRead count{};
  std::array<BackendRead, 3u> terminal{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t expected{1u};
  std::uint32_t state{};
  std::uint32_t outer_iteration{};
  std::uint32_t outer_bound{};
  std::uint32_t inner_iteration{};
  std::uint32_t inner_bound{1u};
  // Number of authored Action transitions represented by this physical
  // occurrence. Scalar Action commands advance one; a proved tile
  // transducer advances the complete inner bound in one device transition.
  std::uint32_t inner_advance{};
  // Fold route 0 consumes the authored seed accumulator, route 1 the first
  // carried bank, and route 2 the second carried bank.
  std::uint32_t route{};
  BackendWindowPhase phase{BackendWindowPhase::Ordinary};
  bool has_terminal{};

  [[nodiscard]] constexpr bool nested() const noexcept {
    return BackendWindowPhaseIsNested(phase);
  }

  [[nodiscard]] constexpr bool advances_outer_state() const noexcept {
    return phase == BackendWindowPhase::Ordinary ||
           phase == BackendWindowPhase::NestedFold;
  }

  // Sole backend-neutral admission authority for one materialized occurrence.
  // Backends may add native resource and transition proofs, but may not
  // reinterpret these bounds, phase routes, or advance rules.
  [[nodiscard]] constexpr bool
  valid_occurrence(const bool transduced_action) const noexcept {
    if (maximum == 0u || tile == 0u || tile > maximum || outer_bound == 0u ||
        outer_iteration >= outer_bound) {
      return false;
    }
    switch (phase) {
    case BackendWindowPhase::Ordinary:
      return !transduced_action;
    case BackendWindowPhase::NestedSeed:
      return !transduced_action && route == 0u && inner_advance == 0u;
    case BackendWindowPhase::NestedAction:
      return inner_bound != 0u && inner_iteration < inner_bound &&
             route == 0u && inner_advance == (transduced_action ? 0u : 1u);
    case BackendWindowPhase::NestedFold:
      return !transduced_action && route < 3u &&
             (inner_advance == 0u || inner_advance == inner_bound);
    }
    return false;
  }

  [[nodiscard]] constexpr bool
  nested_phase(rund::compute::PipelineNestedPhase &out) const noexcept {
    return ProjectBackendWindowPhase(phase, out);
  }
};

// Terminal publication is deliberately outside the authored Program graph.
// Backends select exactly one immutable seed/first/second route from the
// recurrence's ResidentState after canonical Pipeline status has been reduced.
struct BackendPublish final {
  std::array<BackendRead, 3u> sources{};
  BackendRead count{};
  BackendRead target{};
  PreparedKernelPublicationIdentity identity{};
};
