#include "model.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../src/compute/cpu/graph.hpp"
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

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
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

[[nodiscard]] bool CheckNoCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive) {
  using namespace rund::compute::detail;
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != 0u ||
      !std::holds_alternative<std::monostate>(prepared.value())) {
    return false;
  }
  node_compute_allocation::Start();
  const CpuRetainedMemory memory =
      cpu_primitive_scratch_memory(prepared.value());
  node_compute_allocation::Stop();
  return node_compute_allocation::Count() == 0u && memory.host == 0u &&
         memory.tile == 0u;
}

template <class Scratch>
[[nodiscard]] bool CheckOwnedCpuPrimitiveScratch(
    const rund::compute::detail::CpuRuntimePrimitive &primitive,
    const std::uint64_t expected_allocations,
    const std::uint64_t expected_tile_bytes) {
  using namespace rund::compute::detail;
  node_compute_allocation::Start();
  auto prepared = prepare_cpu_scratch(primitive);
  node_compute_allocation::Stop();
  const std::uint64_t allocations = node_compute_allocation::Count();
  if (!prepared || allocations != expected_allocations ||
      cpu_primitive_scratch<Scratch>(prepared.value()) == nullptr) {
    return false;
  }
  node_compute_allocation::Start();
  const CpuRetainedMemory memory =
      cpu_primitive_scratch_memory(prepared.value());
  node_compute_allocation::Stop();
  return node_compute_allocation::Count() == 0u &&
         memory.host == sizeof(Scratch) && memory.tile == expected_tile_bytes;
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
      !CheckOwnedCpuPrimitiveScratch<CpuTransformScratch<Lane>>(
          ScratchPrimitive(Primitive::Transform, transform), 2u,
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
      !CheckOwnedCpuPrimitiveScratch<CpuFactorQrScratch<Lane>>(
          ScratchPrimitive(Primitive::Factor, factor_qr), 4u, 10u * width)) {
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
      !CheckOwnedCpuPrimitiveScratch<CpuSolveQrFactorScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, factor_qr_solve), 2u,
          2u * width)) {
    return 4;
  }
  const SolvePlan matrix_lu = solve_plan(SolveInput::Matrix, FactorOp::LU);
  if (!matrix_lu.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSolveLuScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_lu), 3u,
          matrix_lu.factor_count * width +
              matrix_lu.aux_count * sizeof(rund::kernel::u32))) {
    return 5;
  }
  const SolvePlan matrix_cholesky =
      solve_plan(SolveInput::Matrix, FactorOp::Cholesky);
  if (!matrix_cholesky.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSolveCholeskyScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_cholesky), 2u,
          matrix_cholesky.factor_count * width)) {
    return 6;
  }
  const SolvePlan matrix_qr = solve_plan(SolveInput::Matrix, FactorOp::QR);
  if (!matrix_qr.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSolveQrMatrixScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_qr), 5u,
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
      !CheckOwnedCpuPrimitiveScratch<CpuSpectrumEigenScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, eigen), 4u, 10u * width)) {
    return 8;
  }
  const SpectrumPlan svd_values =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::ValuesOnly, 3u, 2u);
  if (!svd_values.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSpectrumSvdValuesScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_values), 5u,
          10u * width + 2u * sizeof(rund::kernel::u64))) {
    return 9;
  }
  const SpectrumPlan svd_vectors =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Thin, 3u, 2u);
  if (!svd_vectors.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_vectors), 6u,
          16u * width + 2u * sizeof(rund::kernel::u64))) {
    return 10;
  }
  const SpectrumPlan svd_full =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Full, 3u, 2u);
  if (!svd_full.ok ||
      !CheckOwnedCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_full), 6u,
          19u * width + 2u * sizeof(rund::kernel::u64))) {
    return 11;
  }
  return 0;
}

int CheckCpuPrimitiveScratchOwnership() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;

  CpuGraphRun empty_run{};
  node_compute_allocation::Start();
  CpuPrimitiveScratch *const empty_scratch = &cpu_step_scratch(empty_run, 0u);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      empty_scratch != &empty_run.empty_scratch ||
      !std::holds_alternative<std::monostate>(*empty_scratch) ||
      !empty_run.scratch.empty()) {
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
      !CheckOwnedCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u32>>(
          ScratchPrimitive(Primitive::Sort, sort32), 3u,
          5u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 3;
  }
  if (!sort64.ok ||
      !CheckOwnedCpuPrimitiveScratch<
          CpuSortPrimitiveScratch<rund::kernel::u64>>(
          ScratchPrimitive(Primitive::Argsort, sort64), 3u,
          5u * (sizeof(rund::kernel::u64) + sizeof(rund::kernel::u32)))) {
    return 4;
  }

  const ScatterPlan scatter = PlanScatter(ScatterDesc{
      .element = ScatterElement::U32,
      .element_count = 5u,
      .output_count = 8u,
  });
  if (!scatter.ok || scatter.scratch_slots != 16u ||
      !CheckOwnedCpuPrimitiveScratch<CpuScatterPrimitiveScratch>(
          ScratchPrimitive(Primitive::Scatter, scatter), 3u,
          16u * (sizeof(rund::kernel::u32) + sizeof(rund::kernel::u32)))) {
    return 5;
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
        state->cpu->graph == nullptr) {
      return static_cast<CpuCollectiveRun *>(nullptr);
    }
    for (const auto &collective : state->cpu->graph->collectives) {
      if (collective != nullptr) {
        return collective.get();
      }
    }
    return static_cast<CpuCollectiveRun *>(nullptr);
  };
  const CpuCollectiveRun *const scan = active_collective(scan_state);
  const CpuCollectiveRun *const reduce = active_collective(reduce_state);
  if (scan == nullptr || reduce == nullptr || !scan->needs_prefixes ||
      reduce->needs_prefixes || scan->totals.empty() ||
      scan->prefixes.size() != scan->totals.size() || reduce->totals.empty() ||
      !reduce->prefixes.empty() || reduce->prefixes.capacity() != 0u) {
    return 4;
  }
  if (!scan_job->run() || !reduce_job->run()) {
    return 5;
  }
  return 0;
}

} // namespace rund_node_memory_contract
