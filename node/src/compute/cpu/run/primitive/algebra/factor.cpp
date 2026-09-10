#include "local.hpp"

#include "../../../graph.hpp"
#include "../../../scratch.hpp"

#include "../../../../../accel/factor.hpp"

#include <kernel/program/compute/factor/reference.hpp>

namespace rund::compute::detail {

Status run_factor(PrimitiveContext &context) {
  const auto &plan = std::get<kernel::FactorPlan>(context.primitive.plan);
  const bool has_aux = plan.op == kernel::FactorOp::LU;
  const RawCpuBuffer &input = context.port(0u);
  const RawCpuBuffer &factor = context.port(1u);
  const RawCpuBuffer *const aux = has_aux ? &context.port(2u) : nullptr;
  const RawCpuBuffer &status = context.ports.back();
  if (!input || !factor || !status || (aux != nullptr && !*aux)) {
    return finish_cpu_primitive(false, "compute_factor_invalid");
  }
  kernel::FactorResult result{};
  if (plan.element_bytes == sizeof(kernel::i64)) {
    CpuFactorScratchView<kernel::i64> scratch{};
    if (!cpu_factor_scratch(cpu_step_scratch(context.run, context.step), plan,
                            scratch)) {
      return finish_cpu_primitive(false, "compute_factor_invalid");
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
      return finish_cpu_primitive(false, "compute_factor_invalid");
    }
    result = kernel::ReferenceFactorScratchI32(
        reinterpret_cast<const kernel::i32 *>(input.data),
        reinterpret_cast<kernel::i32 *>(factor.data),
        aux == nullptr ? nullptr : reinterpret_cast<kernel::u32 *>(aux->data),
        reinterpret_cast<kernel::u32 *>(status.data), plan, scratch.q,
        scratch.r, scratch.v);
  }
  record_cpu_algebra_status(context.run, Primitive::Factor,
                            result.failed_batches,
                            static_cast<std::uint32_t>(result.first_status));
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
