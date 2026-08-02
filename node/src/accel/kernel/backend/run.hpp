#pragma once

#include "../bindings/build.hpp"
#include "../plan.hpp"
#include "../schedule.hpp"
#include "../scratch.hpp"
#include "../storage.hpp"
#include "../view.hpp"

#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/runtime.hpp>
#include <rund/compute/pipeline/coordinate.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

namespace rund::node::accel::detail {

struct BackendOps;
struct PreparedKernelTemplateRegistry;

struct BoundReset final {
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
  std::uint64_t binding{};
  ExecStep step{};
  ExecStep last{};
  bool external{};
};

using BoundResets = std::vector<BoundReset>;

struct ResetSpan final {
  std::size_t begin{};
  std::size_t count{};

  [[nodiscard]] bool empty() const noexcept { return count == 0u; }
};

using BoundBindings =
    std::variant<StepBinds, ScanBinds, CompactBinds, SegmentedScanBinds,
                 SegmentedReduceBinds, SortBinds, GatherBinds, HistogramBinds,
                 PartitionBinds, ReduceBinds, ScatterBinds, StencilBinds,
                 ScatterReduceBinds, TransformBinds, MatrixBinds, FactorBinds,
                 SolveBinds, SpectrumBinds>;

struct BoundControl final {
  rund::kernel::GraphControl control{};
  const rund::kernel::ResidentBufferRef *count = nullptr;
  const std::shared_ptr<void> *count_handle = nullptr;
  const rund::kernel::ResidentBufferRef *predicate = nullptr;
  const std::shared_ptr<void> *predicate_handle = nullptr;

  [[nodiscard]] bool active() const noexcept {
    return control.has_count() || control.has_predicate();
  }
};

struct BoundStep final {
  std::size_t index = 0u;
  const KernelExecutionStep *step = nullptr;
  const PlannedStep *planned = nullptr;
  const RunBinds *source_binds = nullptr;
  BoundBindings bindings{};
  BoundControl control{};
  DispatchWindowStorage map_windows{};
  ResetSpan resets{};
  bool barrier_before{};
};

inline constexpr std::uint32_t NoNode =
    std::numeric_limits<std::uint32_t>::max();

inline void RecordNode(std::uint32_t *const failed,
                       const BoundStep &step) noexcept {
  if (failed != nullptr && *failed == NoNode && step.step != nullptr) {
    *failed = step.step->source.begin.index;
  }
}

[[nodiscard]] inline bool
BoundStepMatches(const BoundStep &bound,
                 const rund::kernel::NodeKind kind) noexcept {
  return bound.step != nullptr && bound.planned != nullptr &&
         bound.step->kind() == kind;
}

template <typename Active>
[[nodiscard]] inline const Active *
OperationFor(const BoundStep &bound) noexcept {
  return BoundStepMatches(bound, Active::kind)
             ? &bound.step->operation.get<Active>()
             : nullptr;
}

static constexpr std::size_t kInlineBoundStepCapacity = 4u;
using BoundStepStorage = InlineStepStorage<BoundStep, kInlineBoundStepCapacity>;

static_assert(sizeof(BoundStep) <= 512u,
              "canonical backend step exceeded its footprint budget");
static_assert(sizeof(BoundStepStorage) <= 2048u,
              "inline canonical step storage exceeded its footprint budget");

// Exact shared-template route demand frozen by Pipeline preparation before
// the first private backend route is materialized. `owner_count` counts unique
// prepared route owners in one stream and `route_copies` is the public
// generation stride (one ordinary stream or two transactional streams).
// `capacity` is their checked product and therefore sizes the complete shared
// template pool once, independent of which route observes the first miss.
struct BackendTemplateRouteDemand final {
  std::uint32_t owner_count{};
  std::uint32_t route_copies{};
  std::uint32_t capacity{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return owner_count == 0u && route_copies == 0u && capacity == 0u;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return owner_count != 0u &&
           (route_copies == 1u || route_copies == 2u) && capacity != 0u &&
           static_cast<std::uint64_t>(owner_count) * route_copies == capacity;
  }
};

struct BackendRun final {
  const rund::AccelDevice *pick = nullptr;
  const BackendOps *ops = nullptr;
  const KernelExecution *execution = nullptr;
  const BoundResets *resets = nullptr;
  const BoundStep *steps = nullptr;
  std::size_t step_count = 0u;
  std::uint64_t original_dispatch_count = 0u;
  std::uint64_t final_dispatch_count = 0u;
  std::uint64_t *traffic = nullptr;
  const KernelViewLayout *views = nullptr;
  const RunBinds *view_binds = nullptr;
  const KernelScratchLayout *scratch = nullptr;
  // Non-owning cold-preparation cursor. Prepared route resources retain any
  // immutable template owners they acquire from this registry.
  PreparedKernelTemplateRegistry *templates = nullptr;
  // Non-owning scalar cursor valid only during one private route
  // materialization. The cursor clears it before returning to the caller.
  BackendTemplateRouteDemand template_route_demand{};
  std::uint32_t *failed_node = nullptr;
};

struct BackendPublish;
struct BackendWindow;

struct ResidentState final {
  std::uint32_t current{};
  std::uint32_t stopped{};
};

static_assert(std::is_standard_layout_v<ResidentState>);
static_assert(sizeof(ResidentState) == 2u * sizeof(std::uint32_t));
static_assert(alignof(ResidentState) == alignof(std::uint32_t));
static_assert(offsetof(ResidentState, current) == 0u);
static_assert(offsetof(ResidentState, stopped) == 4u);

// A compact nested Pipeline retains one route template for every outer seed,
// inner action, and fold-bank transition. Cold native capture expands those
// templates into physical command occurrences without manufacturing another
// Job/prepared-resource owner per occurrence.
enum class BackendWindowPhase : std::uint8_t {
  Ordinary,
  NestedSeed,
  NestedAction,
  NestedFold,
};

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

// One physical occurrence of a bounded resident recurrence. The three
// terminal routes are the seed, first bank, and second bank. One fixed-width
// ResidentState selector owns the logical transition; an inactive occurrence
// leaves it unchanged and never copies payload between banks.
struct BackendWindow final {
  BackendRead count{};
  std::array<BackendRead, 3u> terminal{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  // Legacy one-dimensional coordinate. For expanded nested occurrences it is
  // the outer coordinate/bound so existing admission remains fail-closed while
  // phase-aware backends consume the explicit two-dimensional coordinates.
  std::uint32_t iteration{};
  std::uint32_t bound{};
  std::uint32_t expected{1u};
  std::uint32_t state{};
  std::uint32_t outer_iteration{};
  std::uint32_t outer_bound{1u};
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
    return phase != BackendWindowPhase::Ordinary;
  }

  [[nodiscard]] constexpr bool advances_outer_state() const noexcept {
    return phase == BackendWindowPhase::Ordinary ||
           phase == BackendWindowPhase::NestedFold;
  }

  [[nodiscard]] constexpr rund::compute::PipelineNestedPhase
  nested_phase() const noexcept {
    switch (phase) {
    case BackendWindowPhase::NestedSeed:
      return rund::compute::PipelineNestedPhase::Seed;
    case BackendWindowPhase::NestedAction:
      return rund::compute::PipelineNestedPhase::Action;
    case BackendWindowPhase::NestedFold:
      return rund::compute::PipelineNestedPhase::Fold;
    case BackendWindowPhase::Ordinary:
      return rund::compute::PipelineNestedPhase::None;
    }
    return rund::compute::PipelineNestedPhase::None;
  }
};

// Terminal publication is deliberately outside the authored Program graph.
// Backends select exactly one immutable seed/first/second route from the
// recurrence's ResidentState after canonical Pipeline status has been reduced.
enum class BackendPublishKind : std::uint8_t {
  Terminal,
  Window,
};

struct BackendPublish final {
  std::array<BackendRead, 3u> sources{};
  std::array<std::uint32_t, 3u> source_ordinals{
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max()};
  BackendRead count{};
  std::uint32_t count_ordinal{std::numeric_limits<std::uint32_t>::max()};
  rund::kernel::ResidentBufferRef target{};
  std::shared_ptr<void> target_handle{};
  std::uint32_t target_ordinal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t final{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  BackendPublishKind kind{BackendPublishKind::Terminal};
};

// Rebuild one already-planned step against an alternate resident binding set.
// Backend View lowering uses this to substitute cold-prepared dense storage
// without creating a second graph, plan, or schedule authority.
[[nodiscard]] bool RebindBoundStep(const BoundStep &source,
                                   const RunBinds &binds, BoundStep &out);

struct BackendBatchEntry final {
  const BackendRun *run = nullptr;
  const std::shared_ptr<void> *prepared = nullptr;
  rund::RuntimeStats *stats = nullptr;
  BackendRecurrence recurrence{};
  // Index into the common compiler's proved TileTransducer table. Backends
  // consume this classification; they do not rediscover it from native state.
  std::uint32_t transducer{std::numeric_limits<std::uint32_t>::max()};
  // Status/resource description is owned once per compact template. Native
  // capture may reference that template many times; occurrence_index is the
  // lexicographic command-order failure key.
  std::uint32_t template_index{};
  std::uint32_t occurrence_index{};
};

struct BoundRun final {
  BackendRun run{};
  BoundStepStorage storage{};
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";

  BoundRun() = default;
  BoundRun(const BoundRun &) = delete;
  BoundRun &operator=(const BoundRun &) = delete;
  BoundRun(BoundRun &&other) noexcept;
  BoundRun &operator=(BoundRun &&other) noexcept;

  void bind(const KernelExecution &execution,
            std::uint64_t original_dispatch_count,
            std::uint64_t final_dispatch_count) noexcept;
};

[[nodiscard]] BoundRun BuildBoundRun(
    const rund::AccelContext &context, const KernelExecution &execution,
    const RunBinds &binds, const BoundResets &resets,
    const PlannedStepStorage &planned, const ScheduledStepOrder &order,
    std::uint64_t original_dispatch_count, std::uint64_t final_dispatch_count);

[[nodiscard]] rund::kernel::BindingSet
MapBindingFor(const BoundStep &step) noexcept;

template <typename Bindings>
[[nodiscard]] const Bindings *BindingsFor(const BoundStep &step) noexcept {
  return std::get_if<Bindings>(&step.bindings);
}

template <typename Bindings>
[[nodiscard]] const Bindings *
BindingsFor(const BoundStep &step, const rund::kernel::NodeKind kind) noexcept {
  return BoundStepMatches(step, kind) ? BindingsFor<Bindings>(step) : nullptr;
}

} // namespace rund::node::accel::detail
