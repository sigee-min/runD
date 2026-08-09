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
#include "../../../job/state.hpp"
#include "../../../program/state.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"
#include "../../graph.hpp"
#include "../../scratch.hpp"

#include <accel/check.hpp>
#include <kernel/program/compute/factor/reference.hpp>
#include <kernel/program/compute/fixed/arithmetic.hpp>
#include <kernel/program/compute/matrix/reference.hpp>
#include <kernel/program/compute/solve/reference.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>
#include <kernel/program/compute/transform/reference.hpp>
#include <kernel/program/compute/window/reference.hpp>
#include <rund/counter.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

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

template <class Lane>
[[nodiscard]] Lane add_window(const Lane left, const Lane right,
                              const kernel::WindowPlan &plan) noexcept {
  if (plan.domain == kernel::ComputeDomain::Fixed) {
    return kernel::compute_fixed_detail::Narrow<Lane>(
        static_cast<kernel::i128>(left) + right, plan.fixed_format.overflow);
  }
  if constexpr (std::is_unsigned_v<Lane>) {
    return static_cast<Lane>(left + right);
  } else {
    using U = std::make_unsigned_t<Lane>;
    return std::bit_cast<Lane>(
        static_cast<U>(std::bit_cast<U>(left) + std::bit_cast<U>(right)));
  }
}

template <class Lane>
[[nodiscard]] Lane subtract_window(const Lane left, const Lane right) noexcept {
  using U = std::make_unsigned_t<Lane>;
  return std::bit_cast<Lane>(
      static_cast<U>(std::bit_cast<U>(left) - std::bit_cast<U>(right)));
}

template <class Lane>
[[nodiscard]] Lane scale_window(const Lane value,
                                const kernel::u64 count) noexcept {
  using U = std::make_unsigned_t<Lane>;
  return std::bit_cast<Lane>(
      static_cast<U>(std::bit_cast<U>(value) * static_cast<U>(count)));
}

template <class Lane>
[[nodiscard]] Lane combine_window(const Lane left, const Lane right,
                                  const kernel::WindowPlan &plan) noexcept {
  if (plan.op == kernel::WindowOp::Min) {
    return std::min(left, right);
  }
  if (plan.op == kernel::WindowOp::Max) {
    return std::max(left, right);
  }
  return add_window(left, right, plan);
}

template <class Lane>
[[nodiscard]] Lane window_identity(const kernel::WindowPlan &plan) noexcept {
  return plan.op == kernel::WindowOp::Min ? std::numeric_limits<Lane>::max()
         : plan.op == kernel::WindowOp::Max
             ? std::numeric_limits<Lane>::lowest()
             : Lane{0};
}

template <class Lane>
[[nodiscard]] kernel::WindowResult
run_prefix_window(const Lane *const input, Lane *const output,
                  const kernel::WindowPlan &semantic,
                  CpuPrimitiveScratch &scratch) noexcept {
  auto *const owner = cpu_primitive_scratch<CpuRangeScratch<Lane>>(scratch);
  if (owner == nullptr || owner->first.size() < semantic.input_count ||
      !owner->second.empty() || semantic.op != kernel::WindowOp::Sum ||
      semantic.fixed_format.overflow == kernel::ComputeOverflow::Saturate) {
    return {};
  }
  std::span<Lane> prefix =
      owner->first.first(static_cast<std::size_t>(semantic.input_count));
  prefix[0u] = input[0u];
  for (kernel::u64 index = 1u; index < semantic.input_count; ++index) {
    prefix[static_cast<std::size_t>(index)] = add_window(
        prefix[static_cast<std::size_t>(index - 1u)], input[index], semantic);
  }
  for (kernel::u64 index = 0u; index < semantic.output_count; ++index) {
    const kernel::u64 anchor = index * semantic.stride;
    const kernel::u64 left =
        anchor < semantic.pad_left ? 0u : anchor - semantic.pad_left;
    const kernel::u64 right_width = semantic.window_size - semantic.pad_left;
    const kernel::u64 right = anchor >= semantic.input_count
                                  ? semantic.input_count - 1u
                              : right_width >= semantic.input_count - anchor
                                  ? semantic.input_count - 1u
                                  : anchor + right_width - 1u;
    Lane value = prefix[static_cast<std::size_t>(right)];
    if (left != 0u) {
      value =
          subtract_window(value, prefix[static_cast<std::size_t>(left - 1u)]);
    }
    if (semantic.boundary == kernel::WindowBoundary::Clamp) {
      const kernel::u64 left_missing =
          anchor < semantic.pad_left ? semantic.pad_left - anchor : 0u;
      const kernel::u64 right_missing =
          anchor >= semantic.input_count
              ? anchor - semantic.input_count + right_width
          : right_width > semantic.input_count - anchor
              ? right_width - (semantic.input_count - anchor)
              : 0u;
      if (left_missing != 0u) {
        value =
            add_window(value, scale_window(input[0u], left_missing), semantic);
      }
      if (right_missing != 0u) {
        value = add_window(
            value,
            scale_window(input[semantic.input_count - 1u], right_missing),
            semantic);
      }
    }
    output[index] = value;
  }
  return {.input_count = semantic.input_count,
          .output_count = semantic.output_count,
          .ok = true,
          .reason = "ok"};
}

template <class Lane>
[[nodiscard]] kernel::WindowResult
run_block_window(const Lane *const input, Lane *const output,
                 const kernel::WindowPlan &semantic,
                 CpuPrimitiveScratch &scratch) noexcept {
  auto *const owner = cpu_primitive_scratch<CpuRangeScratch<Lane>>(scratch);
  kernel::u64 last_anchor = 0u;
  kernel::u64 span = 0u;
  if (owner == nullptr ||
      !kernel::checked::mul(semantic.output_count - 1u, semantic.stride,
                            last_anchor) ||
      !kernel::checked::add(last_anchor, semantic.window_size, span) ||
      owner->first.size() < span || owner->second.size() < span ||
      semantic.op == kernel::WindowOp::Sum) {
    return {};
  }
  const Lane identity = window_identity<Lane>(semantic);
  const auto sample = [&](const kernel::u64 position) noexcept {
    if (position < semantic.pad_left) {
      return semantic.boundary == kernel::WindowBoundary::Clamp ? input[0u]
                                                                : identity;
    }
    const kernel::u64 source = position - semantic.pad_left;
    return source < semantic.input_count ? input[source]
           : semantic.boundary == kernel::WindowBoundary::Clamp
               ? input[semantic.input_count - 1u]
               : identity;
  };
  for (kernel::u64 begin = 0u; begin < span; begin += semantic.window_size) {
    const kernel::u64 end = std::min(span, begin + semantic.window_size);
    for (kernel::u64 index = begin; index < end; ++index) {
      const Lane value = sample(index);
      owner->first[index] =
          index == begin
              ? value
              : combine_window(owner->first[index - 1u], value, semantic);
    }
    for (kernel::u64 cursor = end; cursor > begin; --cursor) {
      const kernel::u64 index = cursor - 1u;
      const Lane value = sample(index);
      owner->second[index] =
          index + 1u == end
              ? value
              : combine_window(value, owner->second[index + 1u], semantic);
    }
  }
  for (kernel::u64 index = 0u; index < semantic.output_count; ++index) {
    const kernel::u64 left = index * semantic.stride;
    const kernel::u64 right = left + semantic.window_size - 1u;
    output[index] =
        combine_window(owner->second[left], owner->first[right], semantic);
  }
  return {.input_count = semantic.input_count,
          .output_count = semantic.output_count,
          .ok = true,
          .reason = "ok"};
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

Status run_window(PrimitiveContext &context) {
  const auto &capacity = std::get<kernel::WindowPlan>(context.primitive.plan);
  if (!context.primitive.range.has_value() || !context.primitive.range->ok()) {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  const auto &range = *context.primitive.range;
  const auto path = range.candidate().disposition();
  const bool resident =
      capacity.count_source != kernel::ComputeCountSource::Descriptor;
  if (resident &&
      (context.job.cpu == nullptr || !context.job.cpu->controlled_count_valid ||
       context.job.cpu->controlled_count > capacity.input_count)) {
    return Status::fail(Reason::WorksetOverflow);
  }
  kernel::WindowPlan plan = capacity;
  if (resident) {
    plan.input_count = context.job.cpu->controlled_count;
    plan.output_count = context.job.cpu->controlled_count;
    plan.input_bytes = plan.input_count * plan.element_bytes;
    plan.output_bytes = plan.output_count * plan.element_bytes;
  }
  if (plan.input_count == 0u) {
    return Status::success();
  }
  const std::size_t output_port = resident ? 2u : 1u;
  if (output_port >= context.ports.size()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  CpuPrimitiveScratch &scratch = cpu_step_scratch(context.run, context.step);
  const auto execute = [&]<class Lane>(const auto reference) {
    const auto *const input =
        reinterpret_cast<const Lane *>(context.port(0u).data);
    auto *const output =
        reinterpret_cast<Lane *>(context.port(output_port).data);
    if (path == node::accel::detail::RangePath::PrefixDifference) {
      return run_prefix_window(input, output, plan, scratch);
    }
    if (path == node::accel::detail::RangePath::BlockPrefixSuffix) {
      return run_block_window(input, output, plan, scratch);
    }
    return path == node::accel::detail::RangePath::Direct
               ? reference(input, output, plan)
               : kernel::WindowResult{};
  };
  kernel::WindowResult result{};
  switch (plan.domain) {
  case kernel::ComputeDomain::I32:
    result =
        execute.template operator()<kernel::i32>(kernel::ReferenceWindowI32);
    break;
  case kernel::ComputeDomain::U32:
    result =
        execute.template operator()<kernel::u32>(kernel::ReferenceWindowU32);
    break;
  case kernel::ComputeDomain::I64:
    result =
        execute.template operator()<kernel::i64>(kernel::ReferenceWindowI64);
    break;
  case kernel::ComputeDomain::U64:
    result =
        execute.template operator()<kernel::u64>(kernel::ReferenceWindowU64);
    break;
  case kernel::ComputeDomain::Fixed:
    result = plan.element_bytes == sizeof(kernel::i64)
                 ? execute.template operator()<kernel::i64>(
                       kernel::ReferenceWindowFixedI64)
                 : execute.template operator()<kernel::i32>(
                       kernel::ReferenceWindowFixedI32);
    break;
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
