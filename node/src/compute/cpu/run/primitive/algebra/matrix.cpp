#include "local.hpp"

#include "../../../graph.hpp"

#include <kernel/program/compute/matrix/reference.hpp>

namespace rund::compute::detail {

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
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
