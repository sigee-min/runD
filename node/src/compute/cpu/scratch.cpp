#include "scratch.hpp"

namespace rund::compute::detail {

Result<CpuPrimitiveScratchPlan>
plan_cpu_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  switch (primitive.kind) {
  case Primitive::Window:
    return plan_cpu_range_scratch(primitive);
  case Primitive::Sort:
  case Primitive::Argsort:
    return plan_cpu_sort_scratch(primitive);
  case Primitive::Scatter:
    return plan_cpu_scatter_scratch(primitive);
  case Primitive::ScatterReduce:
    return plan_cpu_scatter_reduce_scratch(primitive);
  case Primitive::Transform:
    return plan_cpu_transform_scratch(primitive);
  case Primitive::Factor:
    return plan_cpu_factor_scratch(primitive);
  case Primitive::Solve:
    return plan_cpu_solve_scratch(primitive);
  case Primitive::Spectrum:
    return plan_cpu_spectrum_scratch(primitive);
  case Primitive::SegmentedScan:
  case Primitive::SegmentedReduce:
  case Primitive::Compact:
  case Primitive::Gather:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Reduce:
  case Primitive::Stencil:
  case Primitive::Matrix:
    return empty_cpu_scratch_plan();
  }
  return Result<CpuPrimitiveScratchPlan>::fail(Reason::CpuRuntimePlanInvalid);
}

Status append_cpu_primitive_arena_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  switch (scratch.shape) {
  case CpuPrimitiveScratchShape::None:
    return Status::success();
  case CpuPrimitiveScratchShape::RangeI32:
  case CpuPrimitiveScratchShape::RangeU32:
  case CpuPrimitiveScratchShape::RangeI64:
  case CpuPrimitiveScratchShape::RangeU64:
    return append_cpu_range_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::Sort:
    return append_cpu_sort_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::Scatter:
    return append_cpu_scatter_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::ScatterReduce:
    return append_cpu_scatter_reduce_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::Transform:
    return append_cpu_transform_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::FactorQr:
    return append_cpu_factor_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::SolveQrFactor:
  case CpuPrimitiveScratchShape::SolveLu:
  case CpuPrimitiveScratchShape::SolveCholesky:
  case CpuPrimitiveScratchShape::SolveQrMatrix:
    return append_cpu_solve_scratch_plan(arena, scratch);
  case CpuPrimitiveScratchShape::SpectrumEigen:
  case CpuPrimitiveScratchShape::SpectrumSvdValues:
  case CpuPrimitiveScratchShape::SpectrumSvdVectors:
    return append_cpu_spectrum_scratch_plan(arena, scratch);
  }
  return Status::fail(Reason::CpuRuntimePlanInvalid);
}

Result<CpuPrimitiveScratch>
prepare_cpu_scratch(const CpuRuntimePrimitive &primitive,
                    const CpuPrimitiveScratchPlan &request,
                    CpuPreparedArena &arena) {
  const auto planned = plan_cpu_scratch(primitive);
  if (!planned || *planned != request ||
      !scratch_shape_matches(primitive.kind, request.shape)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  switch (request.shape) {
  case CpuPrimitiveScratchShape::None:
    return Result<CpuPrimitiveScratch>::success(CpuPrimitiveScratch{});
  case CpuPrimitiveScratchShape::RangeI32:
  case CpuPrimitiveScratchShape::RangeU32:
  case CpuPrimitiveScratchShape::RangeI64:
  case CpuPrimitiveScratchShape::RangeU64:
    return prepare_cpu_range_scratch(request, arena);
  case CpuPrimitiveScratchShape::Sort:
    return prepare_cpu_sort_scratch(primitive, request, arena);
  case CpuPrimitiveScratchShape::Scatter:
    return prepare_cpu_scatter_scratch(request, arena);
  case CpuPrimitiveScratchShape::ScatterReduce:
    return prepare_cpu_scatter_reduce_scratch(request, arena);
  case CpuPrimitiveScratchShape::Transform:
    return prepare_cpu_transform_scratch(primitive, request, arena);
  case CpuPrimitiveScratchShape::FactorQr:
    return prepare_cpu_factor_scratch(request, arena);
  case CpuPrimitiveScratchShape::SolveQrFactor:
  case CpuPrimitiveScratchShape::SolveLu:
  case CpuPrimitiveScratchShape::SolveCholesky:
  case CpuPrimitiveScratchShape::SolveQrMatrix:
    return prepare_cpu_solve_scratch(request, arena);
  case CpuPrimitiveScratchShape::SpectrumEigen:
  case CpuPrimitiveScratchShape::SpectrumSvdValues:
  case CpuPrimitiveScratchShape::SpectrumSvdVectors:
    return prepare_cpu_spectrum_scratch(request, arena);
  }
  return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
}

} // namespace rund::compute::detail
