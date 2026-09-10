#pragma once

// Included by backend/run.hpp inside rund::node::accel::detail.

// Sealed reset route. Resident identity/extent/usage are retained separately;
// every geometric field in the lookup projection is derived from the one
// constructor-closed Range, so a raw ResidentBufferRef cannot become a second
// reset arithmetic authority after binding.
class BoundReset final {
public:
  [[nodiscard]] static std::optional<BoundReset>
  Seal(const rund::kernel::ResidentBufferRef source,
       std::shared_ptr<void> handle, const reset::Range range,
       const std::uint64_t binding, const ExecStep step, const ExecStep last,
       const bool external) noexcept {
    if (source.id == 0u || source.bytes == 0u ||
        source.usage != rund::kernel::kResidentUsageWrite ||
        handle == nullptr || !range.valid() || range.end() > source.bytes ||
        source.offset_bytes != range.offset() ||
        source.element_bytes != range.element() ||
        source.stride_bytes != range.stride() ||
        source.count != range.count()) {
      return std::nullopt;
    }
    return BoundReset{source, std::move(handle), range, binding, step,
                      last,   external};
  }

  [[nodiscard]] rund::kernel::ResidentBufferRef ref() const noexcept {
    return rund::kernel::ResidentBufferRef{
        .id = resident_,
        .bytes = bytes_,
        .offset_bytes = range_.offset(),
        .element_bytes = range_.element(),
        .stride_bytes = range_.stride(),
        .count = range_.count(),
        .usage = usage_,
    };
  }

  [[nodiscard]] const std::shared_ptr<void> &handle() const noexcept {
    return handle_;
  }

  [[nodiscard]] reset::Range range() const noexcept { return range_; }

  std::uint64_t binding{};
  ExecStep step{};
  ExecStep last{};
  bool external{};

private:
  BoundReset(const rund::kernel::ResidentBufferRef source,
             std::shared_ptr<void> handle, const reset::Range range,
             const std::uint64_t binding, const ExecStep step,
             const ExecStep last, const bool external) noexcept
      : binding{binding}, step{step}, last{last}, external{external},
        resident_{source.id}, bytes_{source.bytes}, usage_{source.usage},
        handle_{std::move(handle)}, range_{range} {}

  std::uint64_t resident_{};
  std::uint64_t bytes_{};
  std::uint32_t usage_{};
  std::shared_ptr<void> handle_{};
  reset::Range range_{};
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
                 PartitionBinds, ReduceBinds, ScatterBinds, RangeBinds,
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
