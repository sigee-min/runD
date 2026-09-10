#include "local.hpp"

#include "../../../graph.hpp"
#include "../../../scratch.hpp"

#include "../../../../../accel/solve.hpp"

#include <kernel/program/compute/solve/reference.hpp>

namespace rund::compute::detail {

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
    return finish_cpu_primitive(false, "compute_solve_invalid");
  }
  kernel::SolveResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuSolveScratchView<kernel::i64> scratch{};
    if (!cpu_solve_scratch(cpu_step_scratch(context.run, context.step), plan,
                           scratch)) {
      return finish_cpu_primitive(false, "compute_solve_invalid");
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
      return finish_cpu_primitive(false, "compute_solve_invalid");
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
  record_cpu_algebra_status(context.run, Primitive::Solve,
                            result.failed_batches,
                            static_cast<std::uint32_t>(result.first_status));
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
