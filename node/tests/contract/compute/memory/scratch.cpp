#include "model.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../src/accel/cpu/scatter/linear.hpp"
#include "../../../../src/accel/kernel/scratch.hpp"
#include "../../../../src/accel/range_aggregate/plan.hpp"
#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/cpu/prepared.hpp"
#include "../../../../src/compute/cpu/scratch.hpp"
#include "../../../../src/compute/job/state.hpp"
#include "../../../../src/compute/memory/cpu.hpp"

#include <kernel/program/compute/compact/plan.hpp>
#include <kernel/program/compute/factor/plan.hpp>
#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/histogram/plan.hpp>
#include <kernel/program/compute/matrix/plan.hpp>
#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/reduce/plan.hpp>
#include <kernel/program/compute/scatter/plan.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>
#include <kernel/program/compute/solve/plan.hpp>
#include <kernel/program/compute/sort/plan.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>
#include <kernel/program/compute/stencil/plan.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace rund_node_memory_contract {

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
ScratchFixedFormat(const std::uint32_t element_bytes) noexcept {
  return rund::kernel::ComputeFixedFormat{
      .integer_bits = 1u,
      .fraction_bits = static_cast<rund::kernel::u8>(element_bytes * 8u - 1u),
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = rund::kernel::ComputeOverflow::Saturate,
      .approximation = rund::kernel::ComputeApproximation::Deterministic,
  };
}

template <class Plan>
[[nodiscard]] rund::compute::detail::CpuRuntimePrimitive
ScratchPrimitive(const rund::compute::detail::Primitive kind,
                 const Plan &plan) {
  rund::compute::detail::CpuRuntimePrimitive primitive{};
  primitive.kind = kind;
  primitive.plan = plan;
  return primitive;
}

[[nodiscard]] rund::compute::detail::CpuRuntimePrimitive
RangeScratchPrimitive(const rund::kernel::WindowPlan &semantic,
                      const rund::node::accel::detail::RangePlan &range) {
  rund::compute::detail::CpuRuntimePrimitive primitive{};
  primitive.kind = rund::compute::detail::Primitive::Window;
  primitive.plan = semantic;
  primitive.range = range;
  return primitive;
}

[[nodiscard]] int CheckAcceleratorScratchPlacementAuthority() {
  using namespace rund::node::accel::detail;
  constexpr KernelScratchRole prefix = KernelScratchRole::from(0u);
  constexpr KernelScratchRole summary = KernelScratchRole::from(1u);
  constexpr KernelScratchRole suffix = KernelScratchRole::from(2u);
  constexpr KernelScratchRole fixup = KernelScratchRole::from(3u);
  constexpr std::array<KernelScratchRequirement, 4u> requirements{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = 24u,
          .alignment = 8u,
          .first_stage = 0u,
          .last_stage = 1u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 16u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = suffix,
          .bytes = 32u,
          .alignment = 8u,
          .first_stage = 1u,
          .last_stage = 2u,
      },
      KernelScratchRequirement{
          .role = fixup,
          .bytes = 8u,
          .alignment = 8u,
          .first_stage = 2u,
          .last_stage = 3u,
      },
  };
  const KernelScratchBatchPlan planned =
      PlanKernelScratchBatch(requirements, 16u, 64u);
  const KernelScratchPlacement *const prefix_place =
      FindKernelScratchPlacement(planned, prefix);
  const KernelScratchPlacement *const summary_place =
      FindKernelScratchPlacement(planned, summary);
  const KernelScratchPlacement *const suffix_place =
      FindKernelScratchPlacement(planned, suffix);
  const KernelScratchPlacement *const fixup_place =
      FindKernelScratchPlacement(planned, fixup);
  KernelScratchPlacement misaligned =
      prefix_place == nullptr ? KernelScratchPlacement{} : *prefix_place;
  misaligned.offset = 8u;
  if (!planned.ok() || std::string_view{planned.reason()} != "ok" ||
      planned.page_count() != 1u || planned.last_bytes() != 64u ||
      planned.backing_bytes() != 64u || planned.payload_bytes() != 56u ||
      planned.placements().size() != requirements.size() ||
      prefix_place == nullptr || prefix_place->page != 0u ||
      prefix_place->offset != 0u || summary_place == nullptr ||
      summary_place->page != 0u || summary_place->offset != 32u ||
      suffix_place == nullptr || suffix_place->page != 0u ||
      suffix_place->offset != 32u || fixup_place == nullptr ||
      fixup_place->page != 0u || fixup_place->offset != 0u ||
      !scratch::valid(*prefix_place, 16u, 64u) ||
      scratch::valid(misaligned, 16u, 64u) ||
      FindKernelScratchPlacement(planned, KernelScratchRole::from(99u)) !=
          nullptr) {
    return 1;
  }

  // Input order is not a second placement authority. Canonical stage/role
  // order produces the exact same frozen placement vector.
  constexpr std::array<KernelScratchRequirement, 4u> permuted{
      requirements[3u], requirements[1u], requirements[0u], requirements[2u]};
  const KernelScratchBatchPlan repeated =
      PlanKernelScratchBatch(permuted, 16u, 64u);
  if (!repeated.ok() || repeated.placements() != planned.placements() ||
      repeated.payload_bytes() != planned.payload_bytes() ||
      repeated.backing_bytes() != planned.backing_bytes() ||
      repeated.last_bytes() != planned.last_bytes() ||
      repeated.page_count() != planned.page_count()) {
    return 2;
  }

  constexpr std::array<KernelScratchRequirement, 2u> concurrent{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = 48u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 32u,
          .alignment = 16u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
  };
  const KernelScratchBatchPlan paged =
      PlanKernelScratchBatch(concurrent, 16u, 64u);
  if (!paged.ok() || paged.page_count() != 2u || paged.last_bytes() != 32u ||
      paged.backing_bytes() != 96u || paged.payload_bytes() != 80u) {
    return 3;
  }
  std::array<KernelScratchRequirement, 2u> serial = concurrent;
  serial[1u].first_stage = 1u;
  serial[1u].last_stage = 1u;
  const KernelScratchBatchPlan reused =
      PlanKernelScratchBatch(serial, 16u, 64u);
  const KernelScratchPlacement *const serial_first =
      FindKernelScratchPlacement(reused, prefix);
  const KernelScratchPlacement *const serial_second =
      FindKernelScratchPlacement(reused, summary);
  if (!reused.ok() || reused.page_count() != 1u || reused.last_bytes() != 48u ||
      reused.backing_bytes() != 48u || reused.payload_bytes() != 48u ||
      serial_first == nullptr || serial_second == nullptr ||
      serial_first->page != serial_second->page ||
      serial_first->offset != serial_second->offset) {
    return 4;
  }

  const KernelScratchBatchPlan empty = PlanKernelScratchBatch({}, 16u, 64u);
  KernelScratchRequirement invalid = requirements[0u];
  invalid.role = {};
  const KernelScratchBatchPlan invalid_role =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.first_stage = 2u;
  invalid.last_stage = 1u;
  const KernelScratchBatchPlan invalid_lifetime =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.alignment = 32u;
  const KernelScratchBatchPlan invalid_alignment =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.alignment = 3u;
  const KernelScratchBatchPlan invalid_power =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.bytes = 0u;
  const KernelScratchBatchPlan invalid_bytes =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  invalid = requirements[0u];
  invalid.bytes = 65u;
  const KernelScratchBatchPlan oversized =
      PlanKernelScratchBatch(std::span{&invalid, 1u}, 16u, 64u);
  std::array<KernelScratchRequirement, 2u> duplicate{requirements[0u],
                                                     requirements[0u]};
  const KernelScratchBatchPlan duplicate_role =
      PlanKernelScratchBatch(duplicate, 16u, 64u);
  if (!empty.ok() || empty.page_count() != 0u || empty.payload_bytes() != 0u ||
      empty.backing_bytes() != 0u || invalid_role.ok() ||
      invalid_lifetime.ok() || invalid_alignment.ok() || invalid_power.ok() ||
      invalid_bytes.ok() || duplicate_role.ok() || oversized.ok() ||
      std::string_view{oversized.reason()} !=
          "compute_resident_bytes_invalid" ||
      PlanKernelScratchBatch(requirements, 0u, 64u).ok() ||
      PlanKernelScratchBatch(requirements, 16u, 63u).ok()) {
    return 5;
  }

  constexpr std::array<KernelScratchRequirement, 2u> overflow{
      KernelScratchRequirement{
          .role = prefix,
          .bytes = std::numeric_limits<std::uint64_t>::max(),
          .alignment = 1u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
      KernelScratchRequirement{
          .role = summary,
          .bytes = 1u,
          .alignment = 1u,
          .first_stage = 0u,
          .last_stage = 0u,
      },
  };
  const KernelScratchBatchPlan overflowed = PlanKernelScratchBatch(
      overflow, 1u, std::numeric_limits<std::uint64_t>::max());
  if (overflowed.ok() ||
      std::string_view{overflowed.reason()} != "compute_pipeline_capacity") {
    return 6;
  }

  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const auto sum_traits =
      RangeTraits::sum_modulo(rund::kernel::ComputeDomain::U32);
  const auto sum_shape =
      sum_traits.has_value()
          ? RangeShape::window(*sum_traits, RangeBoundary::Clamp, 4097u, 4097u,
                               4u)
          : std::nullopt;
  const auto prefix_caps =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                     std::numeric_limits<std::uint32_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(), direct_prefix);
  if (!sum_shape.has_value() || !prefix_caps.has_value()) {
    return 7;
  }
  const RangePlan prefix_plan = PlanRange(*sum_shape, *prefix_caps);
  const KernelScratchBatchPlan prefix_scratch =
      PlanRangeScratch(prefix_plan, 16u, 32768u);
  const KernelScratchBatchPlan prefix_repeated =
      PlanRangeScratch(prefix_plan, 16u, 32768u);
  if (!prefix_plan.ok() ||
      prefix_plan.candidate().disposition() != RangePath::PrefixDifference ||
      !prefix_scratch.ok() || prefix_scratch.page_count() != 1u ||
      prefix_scratch.last_bytes() != 16688u ||
      prefix_scratch.backing_bytes() != 16688u ||
      prefix_scratch.payload_bytes() != 16656u ||
      prefix_scratch.placements() != prefix_repeated.placements() ||
      prefix_scratch.payload_bytes() != prefix_repeated.payload_bytes() ||
      prefix_scratch.backing_bytes() != prefix_repeated.backing_bytes()) {
    return 8;
  }
  for (std::size_t index = 0u; index < prefix_plan.temporary_count(); ++index) {
    const RangeTempReq temporary = prefix_plan.temporary(index);
    const KernelScratchRequirement projected =
        ScratchReqForRangeTemp(temporary);
    const KernelScratchPlacement *const placed =
        FindRangeScratch(prefix_scratch, temporary.role, temporary.ordinal);
    if (!projected.valid() || placed == nullptr ||
        placed->requirement != projected) {
      return 9;
    }
  }
  if (ScratchRoleForRangeTemp(static_cast<RangeTempRole>(255u), 0u).valid() ||
      ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 0u) ==
          ScratchRoleForRangeTemp(RangeTempRole::BlockSummaries, 0u) ||
      ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 0u) ==
          ScratchRoleForRangeTemp(RangeTempRole::PrefixValues, 1u)) {
    return 10;
  }

  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const auto minimum_traits =
      RangeTraits::minimum(rund::kernel::ComputeDomain::U32);
  const auto minimum_shape =
      minimum_traits.has_value()
          ? RangeShape::window(*minimum_traits, RangeBoundary::Clamp, 4097u,
                               4097u, 4u)
          : std::nullopt;
  const auto block_caps =
      RangeCaps::gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                     std::numeric_limits<std::uint32_t>::max(),
                     std::numeric_limits<std::uint32_t>::max(), direct_block);
  if (!minimum_shape.has_value() || !block_caps.has_value()) {
    return 11;
  }
  const RangePlan block_plan = PlanRange(*minimum_shape, *block_caps);
  const KernelScratchBatchPlan block_scratch =
      PlanRangeScratch(block_plan, 16u, 65536u);
  const KernelScratchPlacement *const forward =
      FindRangeScratch(block_scratch, RangeTempRole::ForwardValues, 0u);
  const KernelScratchPlacement *const backward =
      FindRangeScratch(block_scratch, RangeTempRole::BackwardValues, 0u);
  if (!block_plan.ok() ||
      block_plan.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      !block_scratch.ok() || block_scratch.page_count() != 2u ||
      block_scratch.last_bytes() != 49168u ||
      block_scratch.backing_bytes() != 114704u ||
      block_scratch.payload_bytes() != 98328u || forward == nullptr ||
      backward == nullptr || forward->requirement.first_stage != 0u ||
      forward->requirement.last_stage != 1u ||
      backward->requirement.first_stage != 0u ||
      backward->requirement.last_stage != 1u ||
      forward->requirement.role == backward->requirement.role) {
    return 12;
  }

  const RangePlan direct_plan =
      PlanRange(*sum_shape, RangeCaps::cpu_reference());
  const KernelScratchBatchPlan direct_scratch =
      PlanRangeScratch(direct_plan, 16u, 32768u);
  const KernelScratchBatchPlan rejected_scratch =
      PlanRangeScratch(RangePlan::rejected("range_rejected"), 16u, 32768u);
  if (!direct_scratch.ok() || direct_scratch.page_count() != 0u ||
      direct_scratch.payload_bytes() != 0u || rejected_scratch.ok() ||
      std::string_view{rejected_scratch.reason()} != "range_rejected") {
    return 13;
  }
  return 0;
}

[[nodiscard]] bool CheckSharedScatterEpoch() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  const ScatterPlan small_plan = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 3u,
      .output_count = 4u,
  });
  const ScatterPlan large_plan = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 9u,
      .output_count = 16u,
  });
  const CpuRuntimePrimitive small =
      ScratchPrimitive(Primitive::Scatter, small_plan);
  const CpuRuntimePrimitive large =
      ScratchPrimitive(Primitive::Scatter, large_plan);
  const auto small_scratch_plan = plan_cpu_scratch(small);
  const auto large_scratch_plan = plan_cpu_scratch(large);
  if (!small_plan.ok || !large_plan.ok || !small_scratch_plan ||
      !large_scratch_plan ||
      small_scratch_plan->counts[0] >= large_scratch_plan->counts[0]) {
    return false;
  }

  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *small_scratch_plan) ||
      !append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *large_scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  CpuPreparedArenaPlan tampered_segment = arena_plan;
  ++tampered_segment.scatter_marks.offset_bytes;
  const auto rejected_segment = make_cpu_prepared_arena(tampered_segment);
  CpuPreparedArenaPlan tampered_execution = arena_plan;
  --tampered_execution.execution.scatter_slot_count;
  const auto rejected_execution = make_cpu_prepared_arena(tampered_execution);
  if (rejected_segment ||
      rejected_segment.reason() != rund::compute::Reason::CpuRuntimeInvalid ||
      rejected_execution ||
      rejected_execution.reason() != rund::compute::Reason::CpuRuntimeInvalid) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena) {
    return false;
  }
  auto first = prepare_cpu_scratch(small, *small_scratch_plan, **arena);
  auto second = prepare_cpu_scratch(large, *large_scratch_plan, **arena);
  auto *const small_scratch =
      first ? cpu_primitive_scratch<CpuScatterPrimitiveScratch>(*first)
            : nullptr;
  auto *const large_scratch =
      second ? cpu_primitive_scratch<CpuScatterPrimitiveScratch>(*second)
             : nullptr;
  if (small_scratch == nullptr || large_scratch == nullptr ||
      small_scratch->keys.data() != large_scratch->keys.data() ||
      small_scratch->marks.data() != large_scratch->marks.data() ||
      small_scratch->epoch != large_scratch->epoch ||
      small_scratch->mark_capacity.data() !=
          large_scratch->mark_capacity.data() ||
      small_scratch->mark_capacity.size() != large_scratch->marks.size() ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }

  std::fill(small_scratch->mark_capacity.begin(),
            small_scratch->mark_capacity.end(),
            std::numeric_limits<u32>::max());
  *small_scratch->epoch = std::numeric_limits<u32>::max();
  constexpr std::array<u32, 1u> values{7u};
  constexpr std::array<u32, 1u> indices{1u};
  std::array<u32, 16u> small_output{};
  std::array<u32, 16u> large_output{};
  const ScatterResult first_result =
      rund::node::accel::detail::ExecuteLinearScatter(
          *small_scratch, values.data(), indices.data(), small_output.data(),
          values.size(), 4u, small_scratch->marks.size());
  const bool tail_cleared = std::all_of(
      small_scratch->mark_capacity.begin() + small_scratch->marks.size(),
      small_scratch->mark_capacity.end(),
      [](const u32 mark) { return mark == 0u; });
  const ScatterResult second_result =
      rund::node::accel::detail::ExecuteLinearScatter(
          *large_scratch, values.data(), indices.data(), large_output.data(),
          values.size(), 16u, large_scratch->marks.size());
  return first_result.ok && second_result.ok && tail_cleared &&
         small_output[1u] == values[0u] && large_output[1u] == values[0u] &&
         *small_scratch->epoch == 2u;
}

[[nodiscard]] bool CheckNoCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive) {
  using namespace rund::compute::detail;
  const auto scratch_plan = plan_cpu_scratch(primitive);
  if (!scratch_plan) {
    return false;
  }
  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena) {
    return false;
  }
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive, *scratch_plan, **arena);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != 0u ||
      !std::holds_alternative<std::monostate>(prepared.value()) ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }
  return (*arena)->payload_host_bytes() == 0u &&
         (*arena)->payload_tile_bytes() == 0u;
}

template <class Scratch>
[[nodiscard]] bool CheckArenaCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive,
    const std::uint64_t expected_tile_bytes) {
  using namespace rund::compute::detail;
  const auto scratch_plan = plan_cpu_scratch(primitive);
  if (!scratch_plan || scratch_plan->host_bytes != sizeof(Scratch)) {
    return false;
  }
  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *scratch_plan) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return false;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  const std::uint64_t shared_tile_bytes =
      scratch_plan->shape == CpuPrimitiveScratchShape::Scatter
          ? sizeof(rund::kernel::u32)
          : 0u;
  if (!arena || (*arena)->payload_host_bytes() != sizeof(Scratch) ||
      (*arena)->payload_tile_bytes() !=
          expected_tile_bytes + shared_tile_bytes) {
    return false;
  }
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive, *scratch_plan, **arena);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != 0u ||
      cpu_primitive_scratch<Scratch>(prepared.value()) == nullptr ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return false;
  }
  return true;
}

[[nodiscard]] int CheckCpuRangeScratchEnvelope() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  using namespace rund::node::accel::detail;
  constexpr u64 count = 257u;
  constexpr u64 window = 129u;
  constexpr u64 output_count = count - window + 1u;
  const auto sum_traits = RangeTraits::sum_modulo(ComputeDomain::U32);
  const auto min_traits = RangeTraits::minimum(ComputeDomain::U32);
  if (!sum_traits || !min_traits) {
    return 1;
  }
  const auto sum_shape =
      RangeShape::affine(*sum_traits, RangeBoundary::Clip, count, output_count,
                         window, 1u, 0u, sizeof(u32));
  const auto min_shape =
      RangeShape::affine(*min_traits, RangeBoundary::Clip, count, output_count,
                         window, 1u, 0u, sizeof(u32));
  if (!sum_shape || !min_shape) {
    return 2;
  }
  const RangePlan prefix = PlanRange(*sum_shape, RangeCaps::cpu());
  const RangePlan bidirectional = PlanRange(*min_shape, RangeCaps::cpu());
  if (!prefix.ok() || !bidirectional.ok() ||
      prefix.candidate().disposition() != RangePath::PrefixDifference ||
      bidirectional.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      prefix.temporary_count() != 1u || bidirectional.temporary_count() != 2u ||
      prefix.temporary(0u).bytes != count * sizeof(u32) ||
      bidirectional.temporary(0u).bytes != count * sizeof(u32) ||
      bidirectional.temporary(1u).bytes != count * sizeof(u32)) {
    return 3;
  }

  const WindowPlan sum = PlanWindow(WindowDesc{
      .op = WindowOp::Sum,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::U32,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  const WindowPlan minimum = PlanWindow(WindowDesc{
      .op = WindowOp::Min,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::U32,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  const CpuRuntimePrimitive prefix_primitive =
      RangeScratchPrimitive(sum, prefix);
  const CpuRuntimePrimitive bidirectional_primitive =
      RangeScratchPrimitive(minimum, bidirectional);
  const auto prefix_scratch = plan_cpu_scratch(prefix_primitive);
  const auto bidirectional_scratch = plan_cpu_scratch(bidirectional_primitive);
  if (!sum.ok || !minimum.ok || !prefix_scratch || !bidirectional_scratch ||
      prefix_scratch->shape != CpuPrimitiveScratchShape::RangeU32 ||
      bidirectional_scratch->shape != CpuPrimitiveScratchShape::RangeU32 ||
      prefix_scratch->counts[0] != count || prefix_scratch->counts[1] != 0u ||
      bidirectional_scratch->counts[0] != count ||
      bidirectional_scratch->counts[1] != count) {
    return 4;
  }

  CpuPreparedArenaPlan arena_plan{};
  if (!append_cpu_primitive_arena_plan(arena_plan.execution, *prefix_scratch) ||
      !append_cpu_primitive_arena_plan(arena_plan.execution,
                                       *bidirectional_scratch) ||
      arena_plan.execution.primitive_u32_count != 2u * count ||
      arena_plan.execution.primitive_object_payload_bytes !=
          2u * sizeof(CpuRangeScratch<u32>) ||
      !seal_cpu_prepared_arena_plan(arena_plan, 4096u)) {
    return 5;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena || (*arena)->payload_tile_bytes() != 2u * count * sizeof(u32) ||
      (*arena)->payload_host_bytes() != 2u * sizeof(CpuRangeScratch<u32>)) {
    return 6;
  }

  node_compute_allocation::Start();
  auto prefix_prepared =
      prepare_cpu_scratch(prefix_primitive, *prefix_scratch, **arena);
  auto bidirectional_prepared = prepare_cpu_scratch(
      bidirectional_primitive, *bidirectional_scratch, **arena);
  node_compute_allocation::Stop();
  auto *const prefix_owner =
      prefix_prepared
          ? cpu_primitive_scratch<CpuRangeScratch<u32>>(*prefix_prepared)
          : nullptr;
  auto *const bidirectional_owner =
      bidirectional_prepared
          ? cpu_primitive_scratch<CpuRangeScratch<u32>>(*bidirectional_prepared)
          : nullptr;
  if (node_compute_allocation::Count() != 0u || prefix_owner == nullptr ||
      bidirectional_owner == nullptr || prefix_owner->first.size() != count ||
      !prefix_owner->second.empty() ||
      bidirectional_owner->first.size() != count ||
      bidirectional_owner->second.size() != count ||
      prefix_owner->first.data() != bidirectional_owner->first.data() ||
      !(*arena)->claims_complete(arena_plan.execution)) {
    return 7;
  }

  ComputeFixedFormat wrap_format = ScratchFixedFormat(sizeof(i32));
  wrap_format.overflow = ComputeOverflow::Wrap;
  const auto fixed_traits = RangeTraits::sum_modulo(ComputeDomain::Fixed);
  const auto fixed_shape =
      fixed_traits
          ? RangeShape::affine(*fixed_traits, RangeBoundary::Clip, count,
                               output_count, window, 1u, 0u, sizeof(i32))
          : std::nullopt;
  const RangePlan fixed_prefix =
      fixed_shape ? PlanRange(*fixed_shape, RangeCaps::cpu())
                  : RangePlan::rejected("compute_range_shape_invalid");
  const WindowPlan fixed_sum = PlanWindow(WindowDesc{
      .op = WindowOp::Sum,
      .element = WindowElement::U32,
      .boundary = WindowBoundary::Clip,
      .domain = ComputeDomain::Fixed,
      .fixed_format = wrap_format,
      .input_count = count,
      .output_count = output_count,
      .window_size = window,
      .stride = 1u,
      .pad_left = 0u,
  });
  if (!fixed_prefix.ok() || !fixed_sum.ok ||
      fixed_prefix.candidate().disposition() != RangePath::PrefixDifference ||
      !CheckArenaCpuPrimitiveScratch<CpuRangeScratch<i32>>(
          RangeScratchPrimitive(fixed_sum, fixed_prefix),
          count * sizeof(i32))) {
    return 8;
  }
  return 0;
}

template <class Lane> [[nodiscard]] int CheckTypedCpuPrimitiveScratch() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  constexpr std::uint32_t width = sizeof(Lane);
  const ComputeFixedFormat format = ScratchFixedFormat(width);

  const TransformPlan transform = PlanTransform(TransformDesc{
      .op = TransformOp::Fourier,
      .direction = TransformDir::Forward,
      .layout = TransformLayout::Split,
      .normalization = TransformNorm::None,
      .element_count = 8u,
      .fixed_format = format,
  });
  if (!transform.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuTransformScratch<Lane>>(
          ScratchPrimitive(Primitive::Transform, transform),
          transform.workspace_bytes)) {
    return 12;
  }

  const FactorPlan factor_qr = PlanFactor(FactorDesc{
      .op = FactorOp::QR,
      .layout = MatrixLayout::RowMajor,
      .output = FactorOutput::Separate,
      .pivot = PivotOp::None,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 3u,
      .element_bytes = width,
      .fixed_format = format,
  });
  if (!factor_qr.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuFactorQrScratch<Lane>>(
          ScratchPrimitive(Primitive::Factor, factor_qr), 10u * width)) {
    return 1;
  }

  for (const FactorOp op : {FactorOp::LU, FactorOp::Cholesky}) {
    const FactorPlan plan = PlanFactor(FactorDesc{
        .op = op,
        .layout = MatrixLayout::RowMajor,
        .output = FactorOutput::Packed,
        .pivot = op == FactorOp::LU ? PivotOp::Partial : PivotOp::None,
        .rows = 2u,
        .cols = 2u,
        .batch_count = 3u,
        .element_bytes = width,
        .fixed_format = format,
    });
    if (!plan.ok || !CheckNoCpuPrimitiveScratch(
                        ScratchPrimitive(Primitive::Factor, plan))) {
      return 2;
    }
  }

  const auto solve_plan = [&](const SolveInput input, const FactorOp factor) {
    return PlanSolve(SolveDesc{
        .op = SolveOp::Linear,
        .input = input,
        .factor = factor,
        .layout = MatrixLayout::RowMajor,
        .pivot = factor == FactorOp::LU ? PivotOp::Partial : PivotOp::None,
        .rows = 2u,
        .rhs_cols = 1u,
        .batch_count = 3u,
        .element_bytes = width,
        .fixed_format = format,
    });
  };
  for (const FactorOp factor : {FactorOp::LU, FactorOp::Cholesky}) {
    const SolvePlan plan = solve_plan(SolveInput::Factor, factor);
    if (!plan.ok ||
        !CheckNoCpuPrimitiveScratch(ScratchPrimitive(Primitive::Solve, plan))) {
      return 3;
    }
  }
  const SolvePlan factor_qr_solve =
      solve_plan(SolveInput::Factor, FactorOp::QR);
  if (!factor_qr_solve.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveQrFactorScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, factor_qr_solve), 2u * width)) {
    return 4;
  }
  const SolvePlan matrix_lu = solve_plan(SolveInput::Matrix, FactorOp::LU);
  if (!matrix_lu.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveLuScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_lu),
          matrix_lu.factor_count * width +
              matrix_lu.aux_count * sizeof(rund::kernel::u32))) {
    return 5;
  }
  const SolvePlan matrix_cholesky =
      solve_plan(SolveInput::Matrix, FactorOp::Cholesky);
  if (!matrix_cholesky.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveCholeskyScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_cholesky),
          matrix_cholesky.factor_count * width)) {
    return 6;
  }
  const SolvePlan matrix_qr = solve_plan(SolveInput::Matrix, FactorOp::QR);
  if (!matrix_qr.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveQrMatrixScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_qr),
          (matrix_qr.rows * matrix_qr.rhs_cols +
           2u * matrix_qr.rows * matrix_qr.rows + matrix_qr.rows) *
              width)) {
    return 7;
  }

  const auto spectrum_plan =
      [&](const SpectrumOp op, const SpectrumVectors vectors,
          const std::uint64_t rows, const std::uint64_t cols) {
        return PlanSpectrum(SpectrumDesc{
            .op = op,
            .domain = op == SpectrumOp::Eigen ? SpectrumDomain::SymmetricReal
                                              : SpectrumDomain::GeneralReal,
            .vectors = vectors,
            .layout = MatrixLayout::RowMajor,
            .rows = rows,
            .cols = cols,
            .batch_count = 3u,
            .max_iterations = 8u,
            .element_bytes = width,
            .fixed_format = format,
        });
      };
  const SpectrumPlan eigen =
      spectrum_plan(SpectrumOp::Eigen, SpectrumVectors::ValuesOnly, 2u, 2u);
  if (!eigen.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumEigenScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, eigen), 10u * width)) {
    return 8;
  }
  const SpectrumPlan svd_values =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::ValuesOnly, 3u, 2u);
  if (!svd_values.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdValuesScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_values),
          10u * width + 2u * sizeof(rund::kernel::u64))) {
    return 9;
  }
  const SpectrumPlan svd_vectors =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Thin, 3u, 2u);
  if (!svd_vectors.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_vectors),
          16u * width + 2u * sizeof(rund::kernel::u64))) {
    return 10;
  }
  const SpectrumPlan svd_full =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Full, 3u, 2u);
  if (!svd_full.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_full),
          19u * width + 2u * sizeof(rund::kernel::u64))) {
    return 11;
  }
  return 0;
}

int CheckCpuPrimitiveScratchOwnership() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;

  if (const int range = CheckCpuRangeScratchEnvelope(); range != 0) {
    return 80 + range;
  }

  CpuGraphRun empty_run{};
  node_compute_allocation::Start();
  CpuPrimitiveScratch *const empty_scratch = &cpu_step_scratch(empty_run, 0u);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      empty_scratch != &empty_run.empty_scratch ||
      !std::holds_alternative<std::monostate>(*empty_scratch) ||
      empty_run.storage != nullptr) {
    return 1;
  }

  const std::array<CpuRuntimePrimitive, 9u> no_scratch{
      ScratchPrimitive(Primitive::SegmentedScan,
                       PlanSegmentedScan(SegmentedScanDesc{
                           .op = SegmentedScanOp::InclusiveSum,
                           .element = SegmentedScanElement::U32,
                           .element_count = 8u,
                           .block_size = 4u,
                       })),
      ScratchPrimitive(Primitive::SegmentedReduce,
                       PlanSegmentedReduce(SegmentedReduceDesc{
                           .op = ReduceOp::Sum,
                           .element = ReduceElement::U32,
                           .element_count = 8u,
                           .block_size = 4u,
                       })),
      ScratchPrimitive(Primitive::Compact, PlanCompact(CompactDesc{
                                               .element_count = 8u,
                                               .output_capacity = 8u,
                                               .flag_bytes = 4u,
                                               .output_bytes = 4u,
                                           })),
      ScratchPrimitive(Primitive::Gather, PlanGather(GatherDesc{
                                              .element = GatherElement::U32,
                                              .element_count = 8u,
                                              .source_count = 16u,
                                          })),
      ScratchPrimitive(Primitive::Histogram, PlanHistogram(HistogramDesc{
                                                 .index = HistogramIndex::U32,
                                                 .count = HistogramCount::U32,
                                                 .element_count = 8u,
                                                 .bin_count = 4u,
                                             })),
      ScratchPrimitive(Primitive::Partition, PlanPartition(PartitionDesc{
                                                 .element_count = 8u,
                                                 .flag_bytes = 4u,
                                                 .value_bytes = 4u,
                                             })),
      ScratchPrimitive(Primitive::Reduce, PlanReduce(ReduceDesc{
                                              .op = ReduceOp::Sum,
                                              .element = ReduceElement::U32,
                                              .element_count = 8u,
                                              .block_size = 4u,
                                          })),
      ScratchPrimitive(Primitive::Stencil,
                       PlanStencil(StencilDesc{
                           .op = StencilOp::Sum,
                           .element = StencilElement::U32,
                           .boundary = StencilBoundary::Clamp,
                           .element_count = 8u,
                           .radius = 1u,
                       })),
      ScratchPrimitive(Primitive::Matrix,
                       PlanMatrix(MatrixDesc{
                           .op = rund::kernel::MatrixOp::Mul,
                           .layout = MatrixLayout::RowMajor,
                           .arithmetic = MatrixArithmetic::Fixed,
                           .rows = 2u,
                           .cols = 2u,
                           .inner = 2u,
                           .batch_count = 3u,
                           .element_bytes = 4u,
                           .fixed_format = ScratchFixedFormat(4u),
                       })),
  };
  for (std::size_t index = 0u; index < no_scratch.size(); ++index) {
    const CpuRuntimePrimitive &primitive = no_scratch[index];
    const bool valid =
        std::visit([](const auto &plan) { return plan.ok; }, primitive.plan);
    if (!valid) {
      const char *const reason = std::visit(
          [](const auto &plan) { return plan.reason; }, primitive.plan);
      std::fprintf(stderr, "cpu scratch plan invalid index=%zu reason=%s\n",
                   index, reason);
      return 2;
    }
    if (!CheckNoCpuPrimitiveScratch(primitive)) {
      std::fprintf(stderr, "cpu scratch unexpectedly retained index=%zu\n",
                   index);
      return 2;
    }
  }

  const SortPlan sort32 = PlanSort(SortDesc{
      .key = SortKey::U32,
      .value = SortValue::IdentityU32,
      .element_count = 5u,
  });
  const SortPlan sort64 = PlanSort(SortDesc{
      .key = SortKey::U64,
      .value = SortValue::IdentityU32,
      .element_count = 5u,
  });
  if (!sort32.ok ||
      !CheckArenaCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u32>>(
          ScratchPrimitive(Primitive::Sort, sort32),
          5u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 3;
  }
  if (!sort64.ok ||
      !CheckArenaCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u64>>(
          ScratchPrimitive(Primitive::Argsort, sort64),
          5u * (sizeof(rund::kernel::u64) + sizeof(rund::kernel::u32)))) {
    return 4;
  }

  const ScatterPlan scatter = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 5u,
      .output_count = 8u,
  });
  if (!scatter.ok || scatter.scratch_slots != 16u ||
      !CheckArenaCpuPrimitiveScratch<CpuScatterPrimitiveScratch>(
          ScratchPrimitive(Primitive::Scatter, scatter),
          16u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 5;
  }
  if (!CheckSharedScatterEpoch()) {
    return 6;
  }

  if (const int lane32 = CheckTypedCpuPrimitiveScratch<rund::kernel::i32>();
      lane32 != 0) {
    return 10 + lane32;
  }
  if (const int lane64 = CheckTypedCpuPrimitiveScratch<rund::kernel::i64>();
      lane64 != 0) {
    return 30 + lane64;
  }
  return 0;
}

int CheckCpuCollectiveScratchOwnership() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t count = 1024u;
  const std::vector<std::uint32_t> input(count, 1u);
  auto device = open(Target::cpu(2u));
  if (!device) {
    return 1;
  }
  auto scan_program =
      on(*device)
          .input<std::uint32_t>(count)
          .map("collective-scan-input", [](auto value) { return value; })
          .scan(Scan::InclusiveSum)
          .compile();
  auto reduce_program =
      on(*device)
          .input<std::uint32_t>(count)
          .map("collective-reduce-input", [](auto value) { return value; })
          .reduce(Reduce::Sum)
          .compile();
  if (!scan_program || !reduce_program) {
    return 2;
  }
  auto scan_job = scan_program->resident(input);
  auto reduce_job = reduce_program->resident(input);
  if (!scan_job || !reduce_job) {
    return 3;
  }
  const auto scan_state = JobAccess::state(*scan_job);
  const auto reduce_state = JobAccess::state(*reduce_job);
  const auto active_collective = [](const std::shared_ptr<JobState> &state) {
    if (state == nullptr || state->cpu == nullptr ||
        state->cpu->graph == nullptr || state->cpu->graph->storage == nullptr) {
      return static_cast<CpuCollectiveRun *>(nullptr);
    }
    for (auto &collective : state->cpu->graph->storage->collectives) {
      return &collective;
    }
    return static_cast<CpuCollectiveRun *>(nullptr);
  };
  const CpuCollectiveRun *const scan = active_collective(scan_state);
  const CpuCollectiveRun *const reduce = active_collective(reduce_state);
  if (scan == nullptr || reduce == nullptr || !scan->needs_prefixes ||
      reduce->needs_prefixes || scan->totals.empty() ||
      scan->prefixes.size() != scan->totals.size() || reduce->totals.empty() ||
      !reduce->prefixes.empty() || !reduce->prefix_capacity.empty()) {
    return 4;
  }
  if (!scan_job->run() || !reduce_job->run()) {
    return 5;
  }
  return 0;
}

} // namespace rund_node_memory_contract
