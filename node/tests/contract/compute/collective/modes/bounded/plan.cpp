#include "../../../../../../src/compute/cpu/state/program.hpp"
#include "support.hpp"

#include "../../../../../../src/accel/graph/token/local.hpp"
#include "../../../../../../src/compute/cpu/graph.hpp"
#include "../../../../../../src/compute/flow/state.hpp"
#include "../../../../../../src/compute/program/state.hpp"

#include <array>
#include <cstdio>
#include <ranges>
#include <tuple>

namespace rund_node_collective_modes::bounded {
namespace {

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
    const RangePath expected_path =
        operation == 0u ? (boundary == 0u ? RangePath::TiledDifference
                                          : RangePath::PrefixDifference)
                        : RangePath::BlockPrefixSuffix;
    const bool tiled = expected_path == RangePath::TiledDifference;
    const bool fused_block = expected_source == RangeSource::Metal &&
                             expected_path == RangePath::BlockPrefixSuffix;
    if (window.plan.count_source != expected_count ||
        window.plan.input_count != kResidentWindowCapacity ||
        window.plan.output_count != kResidentWindowCapacity ||
        window.plan.window_size != kResidentWindowSize ||
        window.plan.stride != 1u ||
        window.plan.pad_left != kResidentWindowRadius || !range.ok() ||
        range.source_variant() != expected_source ||
        range.candidate().disposition() != expected_path ||
        range.candidate().disposition() == RangePath::SharedHalo ||
        (tiled ? range.stage_count() != 1u : range.stage_count() < 2u) ||
        range.temporary_count() != (tiled         ? 0u
                                    : fused_block ? 1u
                                                  : 2u) ||
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
    if (tiled) {
      if (range.stage(0u).disposition != RangeStageKind::TiledDifference ||
          range.cost().scratch_bytes != 0u ||
          range.stage(0u).groups !=
              (kResidentWindowCapacity +
               range.candidate().width() * kRangeTileOutputsPerLane - 1u) /
                  (range.candidate().width() * kRangeTileOutputsPerLane)) {
        return false;
      }
    } else if (expected_path == RangePath::PrefixDifference) {
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
               (!fused_block &&
                range.temporary(1u).role != RangeTempRole::BackwardValues) ||
               (fused_block &&
                (range.stage(0u).groups != 2u || range.stage(1u).groups != 1u ||
                 range.cost().scratch_bytes !=
                     (kResidentWindowCapacity + kResidentWindowSize - 1u) *
                         sizeof(T)))) {
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
    const auto source_identity = range.source_identity();
    const auto execution_identity = range.execution_identity();
    freeze.source[windows] =
        ResidentRangeIdentity{source_identity.hi, source_identity.lo};
    freeze.execution[windows] =
        ResidentRangeIdentity{execution_identity.hi, execution_identity.lo};
    ++windows;
  }
  return windows == freeze.source.size() &&
         std::ranges::all_of(seen, [](const auto &operation) {
           return operation[0u] && operation[1u];
         });
}

template <class T, class Program>
[[nodiscard]] bool CheckResidentPlanFor(const rund::compute::Backend backend,
                                        const Program &program,
                                        ResidentRangeFreeze &freeze) {
  if (backend == rund::compute::Backend::Cpu) {
    return InspectResidentRangePlans<T>(program, freeze);
  }
  return InspectResidentAccelPlans<T>(program, backend);
}

} // namespace

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const I32Program &program, ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<std::int32_t>(backend, program, freeze);
}

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const U32Program &program, ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<std::uint32_t>(backend, program, freeze);
}

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const I64Program &program, ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<std::int64_t>(backend, program, freeze);
}

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const U64Program &program, ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<std::uint64_t>(backend, program, freeze);
}

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const Fixed32Program &program,
                       ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<rund::compute::Fixed<16, 16>>(backend, program,
                                                            freeze);
}

bool CheckResidentPlan(const rund::compute::Backend backend, const Domain,
                       const Fixed64Program &program,
                       ResidentRangeFreeze &freeze) {
  return CheckResidentPlanFor<rund::compute::Fixed<20, 44>>(backend, program,
                                                            freeze);
}

} // namespace rund_node_collective_modes::bounded
