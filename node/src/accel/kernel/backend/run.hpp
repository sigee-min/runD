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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

namespace rund::node::accel::detail {

struct BackendOps;

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
  std::uint32_t *failed_node = nullptr;
};

struct BackendPublish;
struct BackendWindow;

struct ResidentState final {
  std::uint32_t current{};
  std::uint32_t stopped{};
};

static_assert(sizeof(ResidentState) == 8u);

// Authored recurrence identity is carried from Pipeline planning so a backend
// never guesses that an ordinary ping-pong chain is disposable. A bound of one
// is the canonical non-recurrence value.
struct BackendRecurrence final {
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t bound{1u};
  const BackendWindow *window{};
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
  std::uint32_t iteration{};
  std::uint32_t bound{};
  std::uint32_t expected{1u};
  std::uint32_t state{};
  bool has_terminal{};
};

// Terminal publication is deliberately outside the authored Program graph.
// Backends select exactly one immutable seed/first/second route from the
// recurrence's ResidentState after canonical Pipeline status has been reduced.
struct BackendPublish final {
  std::array<BackendRead, 3u> sources{};
  rund::kernel::ResidentBufferRef target{};
  std::shared_ptr<void> target_handle{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t final{};
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
