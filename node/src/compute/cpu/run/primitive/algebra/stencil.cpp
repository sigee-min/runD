#include "local.hpp"

#include "../../../../../accel/cpu/stencil/reference.hpp"
#include "../../../../type.hpp"
#include "../../../graph.hpp"
#include "../../../state/program.hpp"

namespace rund::compute::detail {

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
  return finish_cpu_primitive(result.ok, result.reason);
}

} // namespace rund::compute::detail
