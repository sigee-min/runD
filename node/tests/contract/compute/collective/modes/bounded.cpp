#include "model.hpp"

#include "../../../../../src/accel/graph/token/local.hpp"
#include "../../../../../src/compute/cpu/graph.hpp"
#include "../../../../../src/compute/flow/state.hpp"
#include "../../../../../src/compute/program/state.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

namespace {

constexpr std::size_t kResidentWindowRadius =
    rund::node::accel::detail::kRangeWidths.back() + 1u;
constexpr std::size_t kResidentWindowCapacity = 2u * kResidentWindowRadius + 1u;
constexpr std::size_t kResidentWindowSize = kResidentWindowCapacity;
constexpr std::size_t kResidentWindowSmallCount = 5u;

template <class T, bool = rund::compute::detail::FixedValue<T>>
struct ResidentBitsOf final {
  using Type = T;
};

template <class T> struct ResidentBitsOf<T, true> final {
  using Type = typename T::Raw;
};

template <class T> using ResidentBits = typename ResidentBitsOf<T>::Type;

template <class T>
using ResidentUnsigned = std::make_unsigned_t<ResidentBits<T>>;

template <class T>
[[nodiscard]] constexpr ResidentUnsigned<T> ResidentRaw(const T value) {
  using Bits = ResidentBits<T>;
  using Unsigned = ResidentUnsigned<T>;
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return static_cast<Unsigned>(value.raw());
  } else if constexpr (std::is_unsigned_v<T>) {
    return value;
  } else {
    return std::bit_cast<Unsigned>(static_cast<Bits>(value));
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentFromRaw(const ResidentUnsigned<T> value) {
  using Bits = ResidentBits<T>;
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::from_raw(std::bit_cast<Bits>(value));
  } else if constexpr (std::is_unsigned_v<T>) {
    return value;
  } else {
    return std::bit_cast<T>(value);
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentWrapAdd(const T left, const T right) {
  return ResidentFromRaw<T>(
      static_cast<ResidentUnsigned<T>>(ResidentRaw(left) + ResidentRaw(right)));
}

template <class T>
[[nodiscard]] constexpr T ResidentPattern(const std::size_t index) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    constexpr std::array<Raw, 8u> pattern{std::numeric_limits<Raw>::max(),
                                          Raw{2},
                                          std::numeric_limits<Raw>::min(),
                                          Raw{-2},
                                          Raw{17},
                                          Raw{-11},
                                          Raw{5},
                                          Raw{-7}};
    return T::from_raw(pattern[index % pattern.size()]);
  } else if constexpr (std::is_unsigned_v<T>) {
    constexpr std::array<T, 8u> pattern{
        std::numeric_limits<T>::max(),
        T{2},
        T{0},
        static_cast<T>(std::numeric_limits<T>::max() - T{1}),
        T{17},
        T{11},
        T{5},
        T{7}};
    return pattern[index % pattern.size()];
  } else {
    constexpr std::array<T, 8u> pattern{std::numeric_limits<T>::max(),
                                        T{2},
                                        std::numeric_limits<T>::min(),
                                        T{-2},
                                        T{17},
                                        T{-11},
                                        T{5},
                                        T{-7}};
    return pattern[index % pattern.size()];
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentPoison(const std::size_t index) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return index % 2u == 0u ? T::max() : T::min();
  } else if constexpr (std::is_unsigned_v<T>) {
    return index % 2u == 0u
               ? std::numeric_limits<T>::max()
               : static_cast<T>(std::numeric_limits<T>::max() - T{1});
  } else {
    return index % 2u == 0u ? std::numeric_limits<T>::max()
                            : std::numeric_limits<T>::min();
  }
}

enum class ResidentWindowOp : std::uint8_t { Sum, Min, Max };

template <class T>
[[nodiscard]] std::vector<T>
ResidentWindowOracle(const std::vector<T> &input, const std::size_t count,
                     const ResidentWindowOp operation, const bool clip) {
  std::vector<T> output;
  output.reserve(count);
  for (std::size_t center = 0u; center < count; ++center) {
    T aggregate = operation == ResidentWindowOp::Sum
                      ? Zero<T>()
                      : (operation == ResidentWindowOp::Min ? Maximum<T>()
                                                            : Minimum<T>());
    for (std::size_t slot = 0u; slot < kResidentWindowSize; ++slot) {
      const std::int64_t logical =
          static_cast<std::int64_t>(center) + static_cast<std::int64_t>(slot) -
          static_cast<std::int64_t>(kResidentWindowRadius);
      std::size_t source = 0u;
      if (logical < 0) {
        if (clip) {
          continue;
        }
      } else if (static_cast<std::uint64_t>(logical) >= count) {
        if (clip) {
          continue;
        }
        source = count - 1u;
      } else {
        source = static_cast<std::size_t>(logical);
      }
      const T value = input[source];
      if (operation == ResidentWindowOp::Sum) {
        aggregate = ResidentWrapAdd(aggregate, value);
      } else if (operation == ResidentWindowOp::Min) {
        aggregate = std::min(aggregate, value);
      } else {
        aggregate = std::max(aggregate, value);
      }
    }
    output.push_back(aggregate);
  }
  return output;
}

template <class T>
[[nodiscard]] std::array<std::vector<T>, 6u>
ResidentWindowOracle(const std::vector<T> &input, const std::size_t count) {
  return {ResidentWindowOracle(input, count, ResidentWindowOp::Sum, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Min, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Max, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Sum, true),
          ResidentWindowOracle(input, count, ResidentWindowOp::Min, true),
          ResidentWindowOracle(input, count, ResidentWindowOp::Max, true)};
}

template <class Output, class T>
[[nodiscard]] bool
ResidentOutputMatches(const Output &output,
                      const std::array<std::vector<T>, 6u> &expected) {
  return output && std::get<0>(*output) == expected[0u] &&
         std::get<1>(*output) == expected[1u] &&
         std::get<2>(*output) == expected[2u] &&
         std::get<3>(*output) == expected[3u] &&
         std::get<4>(*output) == expected[4u] &&
         std::get<5>(*output) == expected[5u];
}

struct ResidentRangeFreeze final {
  std::array<rund::node::accel::detail::RangeIdentity, 6u> source{};
  std::array<rund::node::accel::detail::RangeIdentity, 6u> execution{};

  [[nodiscard]] friend constexpr bool
  operator==(const ResidentRangeFreeze &,
             const ResidentRangeFreeze &) = default;
};

template <class T, class Program>
[[nodiscard]] bool
InspectResidentAccelPlans(const Program &program,
                          const rund::compute::Backend backend) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  static_cast<void>(program);
  static_cast<void>(backend);
  return false;
#else
  using namespace rund::node::accel::detail;
  using namespace rund::kernel;
  const auto &state = rund::compute::detail::FlowAccess::state(program);
  if (!state || !state->accel) {
    return false;
  }
  const std::shared_ptr<KernelToken> token_owner = LookupKernelToken(
      state->accel->kernel.owner, state->accel->kernel.kernel_id);
  const KernelToken *const token = token_owner.get();
  if (backend == rund::compute::Backend::Cpu || token == nullptr ||
      token->kernel_id != state->accel->kernel.kernel_id) {
    return false;
  }
  const RangeSource expected_source = backend == rund::compute::Backend::Metal
                                          ? RangeSource::Metal
                                          : RangeSource::Vulkan;
  const ComputeCountSource expected_count = sizeof(T) == 8u
                                                ? ComputeCountSource::BufferU64
                                                : ComputeCountSource::BufferU32;
  std::array<std::array<bool, 2u>, 3u> seen{};
  std::size_t windows = 0u;
  for (const KernelExecutionStep &step : token->steps) {
    if (step.kind() != NodeKind::Window) {
      continue;
    }
    const operation::Window &window = step.operation.get<operation::Window>();
    const RangePlan &range = window.range;
    const std::size_t operation =
        window.plan.op == WindowOp::Sum
            ? 0u
            : (window.plan.op == WindowOp::Min ? 1u : 2u);
    const std::size_t boundary =
        window.plan.boundary == WindowBoundary::Clamp ? 0u : 1u;
    const RangePath expected_path = operation == 0u
                                        ? RangePath::PrefixDifference
                                        : RangePath::BlockPrefixSuffix;
    if (window.plan.count_source != expected_count ||
        window.plan.input_count != kResidentWindowCapacity ||
        window.plan.output_count != kResidentWindowCapacity ||
        window.plan.window_size != kResidentWindowSize ||
        window.plan.stride != 1u ||
        window.plan.pad_left != kResidentWindowRadius || !range.ok() ||
        range.source_variant() != expected_source ||
        range.candidate().disposition() != expected_path ||
        range.candidate().disposition() == RangePath::SharedHalo ||
        range.stage_count() < 2u || range.temporary_count() != 2u ||
        range.shape().input_count() != kResidentWindowCapacity ||
        range.shape().output_count() != kResidentWindowCapacity ||
        range.shape().window_size() != kResidentWindowSize ||
        range.shape().stride() != 1u ||
        range.shape().padding() != kResidentWindowRadius ||
        !range.shape().resident_counted() || seen[operation][boundary]) {
      std::fprintf(
          stderr,
          "resident accel plan mismatch backend=%u api=%u op=%u boundary=%u "
          "count=%u n=%llu q=%llu k=%llu s=%llu p=%llu source=%u path=%u "
          "width=%u stages=%zu temps=%zu resident=%u\n",
          static_cast<unsigned>(backend), static_cast<unsigned>(token->api),
          static_cast<unsigned>(window.plan.op),
          static_cast<unsigned>(window.plan.boundary),
          static_cast<unsigned>(window.plan.count_source),
          static_cast<unsigned long long>(window.plan.input_count),
          static_cast<unsigned long long>(window.plan.output_count),
          static_cast<unsigned long long>(window.plan.window_size),
          static_cast<unsigned long long>(window.plan.stride),
          static_cast<unsigned long long>(window.plan.pad_left),
          static_cast<unsigned>(range.source_variant()),
          static_cast<unsigned>(range.candidate().disposition()),
          range.candidate().width(), range.stage_count(),
          range.temporary_count(), range.shape().resident_counted() ? 1u : 0u);
      return false;
    }
    if (expected_path == RangePath::PrefixDifference) {
      if (range.stage(0u).disposition != RangeStageKind::PrefixBlock ||
          range.stage(range.stage_count() - 1u).disposition !=
              RangeStageKind::PrefixWindow ||
          range.temporary(0u).role != RangeTempRole::PrefixValues ||
          range.temporary(1u).role != RangeTempRole::BlockSummaries) {
        return false;
      }
    } else if (range.stage_count() != 2u ||
               range.stage(0u).disposition !=
                   RangeStageKind::BlockPrefixSuffix ||
               range.stage(1u).disposition != RangeStageKind::BlockWindow ||
               range.temporary(0u).role != RangeTempRole::ForwardValues ||
               range.temporary(1u).role != RangeTempRole::BackwardValues) {
      return false;
    }
    seen[operation][boundary] = true;
    ++windows;
  }
  const bool complete =
      windows == 6u && std::ranges::all_of(seen, [](const auto &operation) {
        return operation[0u] && operation[1u];
      });
  if (!complete) {
    std::fprintf(stderr,
                 "resident accel plan cardinality backend=%u api=%u "
                 "windows=%zu seen=%u%u/%u%u/%u%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(token->api), windows, seen[0u][0u],
                 seen[0u][1u], seen[1u][0u], seen[1u][1u], seen[2u][0u],
                 seen[2u][1u]);
  }
  return complete;
#endif
}

template <class T, class Program>
[[nodiscard]] bool InspectResidentRangePlans(const Program &program,
                                             ResidentRangeFreeze &freeze) {
  using namespace rund::node::accel::detail;
  using namespace rund::kernel;
  const auto &state = rund::compute::detail::FlowAccess::state(program);
  if (!state || !state->cpu_graph || !state->cpu_graph->runtime) {
    return false;
  }
  const ComputeCountSource expected_count = sizeof(T) == 8u
                                                ? ComputeCountSource::BufferU64
                                                : ComputeCountSource::BufferU32;
  std::array<std::array<bool, 2u>, 3u> seen{};
  std::size_t windows = 0u;
  for (const auto &step : state->cpu_graph->runtime->steps) {
    const auto *primitive =
        std::get_if<rund::compute::detail::CpuRuntimePrimitive>(&step);
    if (primitive == nullptr ||
        primitive->kind != rund::compute::detail::Primitive::Window) {
      continue;
    }
    const auto *window = std::get_if<WindowPlan>(&primitive->plan);
    if (window == nullptr || !primitive->range || !primitive->range->ok() ||
        windows >= freeze.source.size()) {
      return false;
    }
    const RangePlan &range = *primitive->range;
    const RangePath expected_path = window->op == WindowOp::Sum
                                        ? RangePath::PrefixDifference
                                        : RangePath::BlockPrefixSuffix;
    const std::size_t operation = window->op == WindowOp::Sum
                                      ? 0u
                                      : (window->op == WindowOp::Min ? 1u : 2u);
    const std::size_t boundary =
        window->boundary == WindowBoundary::Clamp ? 0u : 1u;
    const bool fixed_wrap =
        !rund::compute::detail::FixedValue<T> ||
        window->fixed_format.overflow == ComputeOverflow::Wrap;
    const std::size_t expected_temporaries =
        expected_path == RangePath::PrefixDifference ? 1u : 2u;
    if (window->count_source != expected_count ||
        window->input_count != kResidentWindowCapacity ||
        window->output_count != kResidentWindowCapacity ||
        window->window_size != kResidentWindowSize || window->stride != 1u ||
        window->pad_left != kResidentWindowRadius || !fixed_wrap ||
        range.source_variant() != RangeSource::Cpu ||
        range.candidate().disposition() != expected_path ||
        range.candidate().width() != kRangeCpuBlockWidth ||
        range.stage_count() != 2u ||
        range.temporary_count() != expected_temporaries ||
        range.shape().input_count() != kResidentWindowCapacity ||
        range.shape().output_count() != kResidentWindowCapacity ||
        range.shape().window_size() != kResidentWindowSize ||
        range.shape().stride() != 1u ||
        range.shape().padding() != kResidentWindowRadius ||
        !range.shape().resident_counted() || seen[operation][boundary]) {
      std::fprintf(
          stderr,
          "resident plan mismatch op=%u boundary=%u count=%u n=%llu q=%llu "
          "k=%llu s=%llu p=%llu source=%u path=%u width=%u stages=%zu "
          "temps=%zu shape=%llu/%llu/%llu/%llu/%llu resident=%u\n",
          static_cast<unsigned>(window->op),
          static_cast<unsigned>(window->boundary),
          static_cast<unsigned>(window->count_source),
          static_cast<unsigned long long>(window->input_count),
          static_cast<unsigned long long>(window->output_count),
          static_cast<unsigned long long>(window->window_size),
          static_cast<unsigned long long>(window->stride),
          static_cast<unsigned long long>(window->pad_left),
          static_cast<unsigned>(range.source_variant()),
          static_cast<unsigned>(range.candidate().disposition()),
          range.candidate().width(), range.stage_count(),
          range.temporary_count(),
          static_cast<unsigned long long>(range.shape().input_count()),
          static_cast<unsigned long long>(range.shape().output_count()),
          static_cast<unsigned long long>(range.shape().window_size()),
          static_cast<unsigned long long>(range.shape().stride()),
          static_cast<unsigned long long>(range.shape().padding()),
          range.shape().resident_counted() ? 1u : 0u);
      return false;
    }
    if (expected_path == RangePath::PrefixDifference) {
      if (range.shape().traits().arithmetic_law() != RangeLaw::ModuloWidth ||
          range.stage(0u).disposition != RangeStageKind::PrefixSequential ||
          range.stage(0u).element_count != kResidentWindowCapacity ||
          range.stage(1u).disposition != RangeStageKind::PrefixWindow ||
          range.stage(1u).element_count != kResidentWindowCapacity ||
          range.temporary(0u).role != RangeTempRole::PrefixValues ||
          range.temporary(0u).bytes != kResidentWindowCapacity * sizeof(T)) {
        return false;
      }
    } else {
      constexpr std::size_t span =
          kResidentWindowCapacity + kResidentWindowSize - 1u;
      if (range.shape().traits().arithmetic_law() != RangeLaw::OrderOnly ||
          range.stage(0u).disposition != RangeStageKind::BlockPrefixSuffix ||
          range.stage(0u).element_count != span ||
          range.stage(1u).disposition != RangeStageKind::BlockWindow ||
          range.stage(1u).element_count != kResidentWindowCapacity ||
          range.temporary(0u).role != RangeTempRole::ForwardValues ||
          range.temporary(1u).role != RangeTempRole::BackwardValues ||
          range.temporary(0u).bytes != span * sizeof(T) ||
          range.temporary(1u).bytes != span * sizeof(T)) {
        return false;
      }
    }
    seen[operation][boundary] = true;
    freeze.source[windows] = range.source_identity();
    freeze.execution[windows] = range.execution_identity();
    ++windows;
  }
  return windows == freeze.source.size() &&
         std::ranges::all_of(seen, [](const auto &operation) {
           return operation[0u] && operation[1u];
         });
}

template <class T>
[[nodiscard]] bool
CheckResidentBoundedWindow(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Count = CountFor<T>;
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<Bounded<T>>(kResidentWindowCapacity)
          .map("resident-window-wrap-source",
               [](auto value) {
                 if constexpr (detail::FixedValue<T>) {
                   return quantize<T, Rounding::NearestEven, Overflow::Wrap>(
                       value);
                 } else {
                   return value;
                 }
               })
          .branch([](auto values) {
            return outputs(values.window({.op = Window::Sum,
                                          .radius = kResidentWindowRadius}),
                           values.window({.op = Window::Min,
                                          .radius = kResidentWindowRadius}),
                           values.window({.op = Window::Max,
                                          .radius = kResidentWindowRadius}),
                           values.window({.op = Window::Sum,
                                          .radius = kResidentWindowRadius,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Min,
                                          .radius = kResidentWindowRadius,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Max,
                                          .radius = kResidentWindowRadius,
                                          .edge = WindowEdge::Clip}));
          })
          .compile();
  if (!program) {
    std::fprintf(stderr,
                 "compute resident bounded window compile width=%zu fixed=%u "
                 "reason=%.*s\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }

  ResidentRangeFreeze initial_freeze{};
  if (backend == Backend::Cpu &&
      !InspectResidentRangePlans<T>(*program, initial_freeze)) {
    std::fprintf(stderr,
                 "compute resident bounded window frozen plan width=%zu "
                 "fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }
  if (backend != Backend::Cpu &&
      !InspectResidentAccelPlans<T>(*program, backend)) {
    std::fprintf(stderr,
                 "compute resident bounded window accel plan width=%zu "
                 "fixed=%u backend=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<unsigned>(backend));
    return false;
  }

  std::vector<T> full(kResidentWindowCapacity);
  std::vector<T> small(kResidentWindowCapacity);
  std::vector<T> zero(kResidentWindowCapacity, Zero<T>());
  for (std::size_t index = 0u; index < kResidentWindowCapacity; ++index) {
    full[index] = ResidentPattern<T>(index);
    small[index] = index < kResidentWindowSmallCount
                       ? ResidentPattern<T>(index + 3u)
                       : ResidentPoison<T>(index);
  }
  const std::array<Count, 1u> full_count{
      static_cast<Count>(kResidentWindowCapacity)};
  const std::array<Count, 1u> small_count{
      static_cast<Count>(kResidentWindowSmallCount)};
  const std::array<Count, 1u> zero_count{Count{0}};
  const std::array<Count, 1u> overflow_count{
      static_cast<Count>(kResidentWindowCapacity + 1u)};

  auto job = program->resident(full, full_count);
  if (!job || !job->run() ||
      !ResidentOutputMatches(job->read_all(),
                             ResidentWindowOracle(full, full.size())) ||
      !job->write(small, small_count) || !job->run()) {
    std::fprintf(stderr,
                 "compute resident bounded window full/small execution "
                 "width=%zu fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }
  const auto small_expected =
      ResidentWindowOracle(small, kResidentWindowSmallCount);
  auto small_output = job->read_all();
  if (!ResidentOutputMatches(small_output, small_expected)) {
    std::fprintf(stderr,
                 "compute resident bounded window poisoned-tail mismatch "
                 "width=%zu fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }

  const Status overflow = job->write(full, overflow_count);
  auto retained = job->read_all();
  if (overflow || overflow.error() != "compute_workset_overflow" ||
      !ResidentOutputMatches(retained, small_expected) ||
      !job->write(zero, zero_count) || !job->run()) {
    std::fprintf(stderr,
                 "compute resident bounded window overflow/zero width=%zu "
                 "fixed=%u reason=%.*s\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<int>(overflow.error().size()),
                 overflow.error().data());
    return false;
  }
  const auto empty_expected = ResidentWindowOracle(zero, 0u);
  if (!ResidentOutputMatches(job->read_all(), empty_expected) ||
      !job->write(full, full_count) || !job->run() ||
      !ResidentOutputMatches(job->read_all(),
                             ResidentWindowOracle(full, full.size()))) {
    std::fprintf(stderr,
                 "compute resident bounded window zero/retry width=%zu "
                 "fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }

  ResidentRangeFreeze final_freeze{};
  return backend != Backend::Cpu ||
         (InspectResidentRangePlans<T>(*program, final_freeze) &&
          initial_freeze == final_freeze);
}

} // namespace

template <class T>
[[nodiscard]] bool CheckBounded(const rund::compute::Backend backend,
                                DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template map<T>("mode-bounded-input", input.size(),
                           [](auto value) { return Store<T>(value); })
          .filter([](auto value) { return value != Zero<T>(); })
          .branch([](auto values) {
            const auto maximum = values.reduce(Reduce::Max);
            return outputs(
                values.map("mode-bounded-map",
                           [](auto value) { return Store<T>(value); }),
                values.scan(Scan::InclusiveSum),
                values.scan(Scan::ExclusiveSum), values.reduce(Reduce::Sum),
                values.reduce(Reduce::Min), maximum, values.sort(),
                values.argsort(), values.count(),
                maximum.map("mode-scalar-map",
                            [](auto value) { return Store<T>(value); }));
          })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "compute modes compile backend=%u family=bounded reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  return job && SameSuccess(*job, backend, "bounded", evidence.bounded);
}

template <class T>
[[nodiscard]] bool CheckBoundedWindow(const rund::compute::Backend backend,
                                      DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template map<T>("mode-bounded-window-input", input.size(),
                           [](auto value) { return Store<T>(value); })
          .filter([](auto value) { return value != Zero<T>(); })
          .branch([](auto values) {
            return outputs(values.window({.op = Window::Sum, .radius = 1u}),
                           values.window({.op = Window::Min, .radius = 1u}),
                           values.window({.op = Window::Max, .radius = 1u}),
                           values.window({.op = Window::Sum,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Min,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}),
                           values.window({.op = Window::Max,
                                          .radius = 1u,
                                          .edge = WindowEdge::Clip}));
          })
          .compile();
  if (!program) {
    std::fprintf(
        stderr,
        "compute modes compile backend=%u family=bounded-window reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job) {
    return false;
  }
  const bool host_same =
      SameSuccess(*job, backend, "bounded-window", evidence.bounded_window);
  std::vector<std::int64_t> logical;
  logical.reserve(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const std::int64_t value = TailInteger<T>(index);
    if (value != 0) {
      logical.push_back(value);
    }
  }
  const auto clamped = ExpectedWindows<T>(logical, false);
  const auto clipped = ExpectedWindows<T>(logical, true);
  auto output = job->read_all();
  const std::array<bool, 6u> fields{
      output && std::get<0>(*output) == clamped[0u],
      output && std::get<1>(*output) == clamped[1u],
      output && std::get<2>(*output) == clamped[2u],
      output && std::get<3>(*output) == clipped[0u],
      output && std::get<4>(*output) == clipped[1u],
      output && std::get<5>(*output) == clipped[2u]};
  const bool same =
      std::ranges::all_of(fields, [](const bool value) { return value; });
  if (!same) {
    std::fprintf(stderr,
                 "compute modes bounded window golden mismatch backend=%u "
                 "width=%zu fields=%u%u%u%u%u%u\n",
                 static_cast<unsigned>(backend), sizeof(T), fields[0u],
                 fields[1u], fields[2u], fields[3u], fields[4u], fields[5u]);
  }
  return host_same && same;
}

template <class T>
[[nodiscard]] bool CheckBoundedDomain(const rund::compute::Backend backend,
                                      DomainEvidence &evidence) {
  return CheckBounded<T>(backend, evidence) &&
         CheckBoundedWindow<T>(backend, evidence) &&
         CheckResidentBoundedWindow<T>(backend);
}

[[nodiscard]] bool CheckBounded(const rund::compute::Backend backend,
                                DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckBoundedDomain<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckBoundedDomain<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckBoundedDomain<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckBoundedDomain<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckBoundedDomain<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckBoundedDomain<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes
