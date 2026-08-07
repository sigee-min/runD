#include "algebra.hpp"

#include "../../../../accel/cpu/stencil/reference.hpp"
#include "../../../../accel/factor.hpp"
#include "../../../../accel/matrix.hpp"
#include "../../../../accel/solve.hpp"
#include "../../../../accel/spectrum.hpp"
#include "../../../../accel/stencil.hpp"
#include "../../../../accel/transform.hpp"
#include "../../../backend.hpp"
#include "../../../host.hpp"
#include "../../../program/state.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"
#include "../../graph.hpp"
#include "../../scratch.hpp"

#include <accel/check.hpp>
#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/matrix/reference.hpp>
#include <kernel/program/compute/solve/reference.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>
#include <kernel/program/compute/transform/reference.hpp>
#include <rund/counter.hpp>

#include <cstdint>

namespace rund::compute::detail {

namespace {

[[nodiscard]] Status finish(const rund::AccelCheck check) noexcept {
  return check.ok ? Status::success()
                  : Status::fail(project_reason(
                        check.reason, Reason::PrimitiveBackendFailed));
}

void record_status(CpuGraphRun &run, const Primitive primitive,
                   const std::uint64_t failed,
                   const std::uint32_t first) noexcept {
  if (failed == 0u || first == 0u) {
    return;
  }
  run.semantic_failure_count = ::rund::detail::counter::SaturatingAdd(
      run.semantic_failure_count, failed);
  if (run.semantic_status == 0u) {
    run.semantic_primitive = primitive;
    run.semantic_status = first;
  }
}

} // namespace

Status run_stencil(PrimitiveContext &context) {
  const auto &primitive = context.primitive;
  const auto &plan = std::get<kernel::StencilPlan>(primitive.plan);
  const Type type =
      context.cpu.runtime->values[primitive.inputs.front() - 1u].type;
  const kernel::ComputeDomain domain = type_domain(type);
  const bool signed_domain = domain == kernel::ComputeDomain::I32 ||
                             domain == kernel::ComputeDomain::I64 ||
                             domain == kernel::ComputeDomain::Fixed;
  kernel::StencilResult result{};
  if (signed_domain && plan.element == kernel::StencilElement::U32) {
    result = node::accel::detail::ExecuteSignedStencilReference(
        plan.op, reinterpret_cast<const kernel::i32 *>(context.port(0u).data),
        reinterpret_cast<kernel::i32 *>(context.port(1u).data),
        plan.element_count, plan.radius);
  } else if (signed_domain) {
    result = node::accel::detail::ExecuteSignedStencilReference(
        plan.op, reinterpret_cast<const kernel::i64 *>(context.port(0u).data),
        reinterpret_cast<kernel::i64 *>(context.port(1u).data),
        plan.element_count, plan.radius);
  } else if (plan.element == kernel::StencilElement::U32) {
    result = node::accel::detail::ExecuteStencilReference(
        plan.op, reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
        reinterpret_cast<kernel::u32 *>(context.port(1u).data),
        plan.element_count, plan.radius);
  } else {
    result = node::accel::detail::ExecuteStencilReference(
        plan.op, reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
        reinterpret_cast<kernel::u64 *>(context.port(1u).data),
        plan.element_count, plan.radius);
  }
  return finish({result.ok, result.reason});
}

Status run_transform(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::TransformPlan>(context.primitive.plan);
  CpuPrimitiveScratch &scratch = cpu_step_scratch(context.run, context.step);
  const auto result =
      context.port(0u).bytes / plan.element_count == sizeof(kernel::i64)
          ? kernel::ReferenceFourierSplitI64(
                reinterpret_cast<const kernel::i64 *>(context.port(0u).data),
                reinterpret_cast<const kernel::i64 *>(context.port(1u).data),
                reinterpret_cast<kernel::i64 *>(context.port(2u).data),
                reinterpret_cast<kernel::i64 *>(context.port(3u).data), plan,
                cpu_transform_twiddle<kernel::i64>(scratch))
          : kernel::ReferenceFourierSplitI32(
                reinterpret_cast<const kernel::i32 *>(context.port(0u).data),
                reinterpret_cast<const kernel::i32 *>(context.port(1u).data),
                reinterpret_cast<kernel::i32 *>(context.port(2u).data),
                reinterpret_cast<kernel::i32 *>(context.port(3u).data), plan,
                cpu_transform_twiddle<kernel::i32>(scratch));
  return finish({result.ok, result.reason});
}

Status run_matrix(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::MatrixPlan>(context.primitive.plan);
  const std::size_t output = context.ports.size() - 1u;
  kernel::MatrixResult result{};
  if (plan.op == kernel::MatrixOp::Transpose) {
    result =
        plan.element_bytes == sizeof(kernel::i64)
            ? kernel::ReferenceMatrixTransposeI64(
                  reinterpret_cast<const kernel::i64 *>(context.port(0u).data),
                  reinterpret_cast<kernel::i64 *>(context.port(output).data),
                  plan)
            : kernel::ReferenceMatrixTransposeI32(
                  reinterpret_cast<const kernel::i32 *>(context.port(0u).data),
                  reinterpret_cast<kernel::i32 *>(context.port(output).data),
                  plan);
  } else {
    result =
        plan.element_bytes == sizeof(kernel::i64)
            ? kernel::ReferenceMatrixMulI64(
                  reinterpret_cast<const kernel::i64 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::i64 *>(context.port(1u).data),
                  reinterpret_cast<kernel::i64 *>(context.port(output).data),
                  plan)
            : kernel::ReferenceMatrixMulI32(
                  reinterpret_cast<const kernel::i32 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::i32 *>(context.port(1u).data),
                  reinterpret_cast<kernel::i32 *>(context.port(output).data),
                  plan);
  }
  return finish({result.ok, result.reason});
}

Status run_factor(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::FactorPlan>(context.primitive.plan);
  const bool has_aux = plan.op == kernel::FactorOp::LU;
  const RawCpuBuffer &input = context.port(0u);
  const RawCpuBuffer &factor = context.port(1u);
  const RawCpuBuffer *const aux = has_aux ? &context.port(2u) : nullptr;
  const RawCpuBuffer &status = context.ports.back();
  if (!input || !factor || !status || (aux != nullptr && !*aux)) {
    return finish({false, "compute_factor_invalid"});
  }
  kernel::FactorResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuFactorScratchView<kernel::i64> scratch{};
    if (!cpu_factor_scratch(cpu_step_scratch(context.run, context.step), plan,
                            scratch)) {
      return finish({false, "compute_factor_invalid"});
    }
    result = kernel::ReferenceFactorScratchI64(
        reinterpret_cast<const kernel::i64 *>(input.data),
        reinterpret_cast<kernel::i64 *>(factor.data),
        aux == nullptr ? nullptr : reinterpret_cast<kernel::u32 *>(aux->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.q,
        scratch.r, scratch.v);
  } else {
    CpuFactorScratchView<kernel::i32> scratch{};
    if (!cpu_factor_scratch(cpu_step_scratch(context.run, context.step), plan,
                            scratch)) {
      return finish({false, "compute_factor_invalid"});
    }
    result = kernel::ReferenceFactorScratchI32(
        reinterpret_cast<const kernel::i32 *>(input.data),
        reinterpret_cast<kernel::i32 *>(factor.data),
        aux == nullptr ? nullptr : reinterpret_cast<kernel::u32 *>(aux->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.q,
        scratch.r, scratch.v);
  }
  record_status(context.run, Primitive::Factor, result.failed_batches,
                static_cast<std::uint32_t>(result.first_status));
  return finish({result.ok, result.reason});
}

Status run_solve(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::SolvePlan>(context.primitive.plan);
  const bool has_aux = plan.input == kernel::SolveInput::Factor &&
                       plan.factor == kernel::FactorOp::LU;
  const RawCpuBuffer &primary = context.port(0u);
  const RawCpuBuffer *const aux = has_aux ? &context.port(1u) : nullptr;
  const RawCpuBuffer &rhs = context.port(has_aux ? 2u : 1u);
  const RawCpuBuffer &output = context.port(has_aux ? 3u : 2u);
  const RawCpuBuffer &status = context.port(has_aux ? 4u : 3u);
  if (!primary || !rhs || !output || !status || (aux != nullptr && !*aux)) {
    return finish({false, "compute_solve_invalid"});
  }
  kernel::SolveResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuSolveScratchView<kernel::i64> scratch{};
    if (!cpu_solve_scratch(cpu_step_scratch(context.run, context.step), plan,
                           scratch)) {
      return finish({false, "compute_solve_invalid"});
    }
    result = kernel::ReferenceSolveScratchI64(
        reinterpret_cast<const kernel::i64 *>(primary.data),
        aux == nullptr ? nullptr
                       : reinterpret_cast<const kernel::u32 *>(aux->data),
        reinterpret_cast<const kernel::i64 *>(rhs.data),
        reinterpret_cast<kernel::i64 *>(output.data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.factor,
        scratch.pivots, scratch.y, scratch.q, scratch.r, scratch.v);
  } else {
    CpuSolveScratchView<kernel::i32> scratch{};
    if (!cpu_solve_scratch(cpu_step_scratch(context.run, context.step), plan,
                           scratch)) {
      return finish({false, "compute_solve_invalid"});
    }
    result = kernel::ReferenceSolveScratchI32(
        reinterpret_cast<const kernel::i32 *>(primary.data),
        aux == nullptr ? nullptr
                       : reinterpret_cast<const kernel::u32 *>(aux->data),
        reinterpret_cast<const kernel::i32 *>(rhs.data),
        reinterpret_cast<kernel::i32 *>(output.data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.factor,
        scratch.pivots, scratch.y, scratch.q, scratch.r, scratch.v);
  }
  record_status(context.run, Primitive::Solve, result.failed_batches,
                static_cast<std::uint32_t>(result.first_status));
  return finish({result.ok, result.reason});
}

Status run_spectrum(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::SpectrumPlan>(context.primitive.plan);
  const bool has_vectors = plan.vector_count != 0u;
  const RawCpuBuffer &input = context.port(0u);
  const RawCpuBuffer &values = context.port(1u);
  const RawCpuBuffer *const vectors = has_vectors ? &context.port(2u) : nullptr;
  const RawCpuBuffer &status = context.ports.back();
  if (!input || !values || !status || (vectors != nullptr && !*vectors)) {
    return finish({false, "compute_spectrum_invalid"});
  }
  kernel::SpectrumResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuSpectrumScratchView<kernel::i64> scratch{};
    if (!cpu_spectrum_scratch(cpu_step_scratch(context.run, context.step), plan,
                              scratch)) {
      return finish({false, "compute_spectrum_invalid"});
    }
    result = kernel::ReferenceSpectrumScratchI64(
        reinterpret_cast<const kernel::i64 *>(input.data),
        reinterpret_cast<kernel::i64 *>(values.data),
        vectors == nullptr ? nullptr
                           : reinterpret_cast<kernel::i64 *>(vectors->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.matrix,
        scratch.vectors, scratch.values, scratch.order, scratch.u);
  } else {
    CpuSpectrumScratchView<kernel::i32> scratch{};
    if (!cpu_spectrum_scratch(cpu_step_scratch(context.run, context.step), plan,
                              scratch)) {
      return finish({false, "compute_spectrum_invalid"});
    }
    result = kernel::ReferenceSpectrumScratchI32(
        reinterpret_cast<const kernel::i32 *>(input.data),
        reinterpret_cast<kernel::i32 *>(values.data),
        vectors == nullptr ? nullptr
                           : reinterpret_cast<kernel::i32 *>(vectors->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.matrix,
        scratch.vectors, scratch.values, scratch.order, scratch.u);
  }
  record_status(context.run, Primitive::Spectrum, result.failed_batches,
                static_cast<std::uint32_t>(result.first_status));
  return finish({result.ok, result.reason});
}

} // namespace rund::compute::detail
