#include "local.hpp"

#include "../../../scratch.hpp"

#include <kernel/program/compute/transform/reference.hpp>

namespace rund::compute::detail {

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
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
