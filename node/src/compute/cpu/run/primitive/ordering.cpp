#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../type.hpp"
#include "../../graph.hpp"
#include "../../scratch.hpp"

#include "../../../../accel/cpu/scatter/linear.hpp"
#include "../../../../accel/cpu/sort/radix.hpp"

#include <cstdint>

namespace rund::compute::detail {

Status run_cpu_primitive_ordering(PrimitiveContext &context) {
  if (context.primitive.kind == Primitive::Sort ||
      context.primitive.kind == Primitive::Argsort) {
    const Type type =
        context.cpu.runtime->values[context.primitive.inputs.front() - 1u].type;
    const kernel::ComputeDomain domain = type_domain(type);
    const bool signed_order = domain == kernel::ComputeDomain::I32 ||
                              domain == kernel::ComputeDomain::I64 ||
                              domain == kernel::ComputeDomain::Fixed;
    const auto &plan = std::get<kernel::SortPlan>(context.primitive.plan);
    kernel::u32 active_count = static_cast<kernel::u32>(plan.element_count);
    const bool bounded = context.primitive.inputs.size() == 2u;
    if (bounded) {
      std::uint32_t count = 0u;
      const Status count_status = read_cpu_primitive_count(
          context, context.primitive.inputs[1u], plan.element_count, count);
      if (!count_status) {
        return Status::fail(count_status.reason());
      }
      active_count = static_cast<kernel::u32>(count);
    }
    const std::size_t key_output = bounded ? 2u : 1u;
    const std::size_t value_output = key_output + 1u;
    if (plan.key == kernel::SortKey::U32) {
      auto *const scratch =
          cpu_primitive_scratch<CpuSortPrimitiveScratch<kernel::u32>>(
              cpu_step_scratch(context.run, context.step));
      const auto result =
          scratch == nullptr
              ? rund::AccelCheck{false, "compute_sort_invalid"}
              : node::accel::detail::ExecuteCpuRadixSortPrepared(
                    reinterpret_cast<const kernel::u32 *>(
                        context.port(0u).data),
                    nullptr,
                    reinterpret_cast<kernel::u32 *>(
                        context.port(key_output).data),
                    reinterpret_cast<kernel::u32 *>(
                        context.port(value_output).data),
                    active_count, plan.radix_pass_count, true, signed_order,
                    scratch->keys.data(), scratch->values.data(),
                    scratch->counts, scratch->offsets);
      return finish_cpu_primitive(result.ok, result.reason);
    }
    auto *const scratch =
        cpu_primitive_scratch<CpuSortPrimitiveScratch<kernel::u64>>(
            cpu_step_scratch(context.run, context.step));
    const auto result =
        scratch == nullptr
            ? rund::AccelCheck{false, "compute_sort_invalid"}
            : node::accel::detail::ExecuteCpuRadixSortPrepared(
                  reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
                  nullptr,
                  reinterpret_cast<kernel::u64 *>(
                      context.port(key_output).data),
                  reinterpret_cast<kernel::u32 *>(
                      context.port(value_output).data),
                  active_count, plan.radix_pass_count, true, signed_order,
                  scratch->keys.data(), scratch->values.data(), scratch->counts,
                  scratch->offsets);
    return finish_cpu_primitive(result.ok, result.reason);
  }

  if (context.primitive.kind == Primitive::Scatter) {
    const auto &plan = std::get<kernel::ScatterPlan>(context.primitive.plan);
    CpuScatterPrimitiveScratch *const scratch =
        cpu_primitive_scratch<CpuScatterPrimitiveScratch>(
            cpu_step_scratch(context.run, context.step));
    if (scratch == nullptr) {
      return finish_cpu_primitive(false, "compute_scatter_invalid");
    }
    const auto result =
        plan.element == kernel::ScatterElement::U32
            ? node::accel::detail::ExecuteLinearScatter(
                  *scratch,
                  reinterpret_cast<const kernel::u32 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  reinterpret_cast<kernel::u32 *>(context.port(2u).data),
                  plan.element_count, plan.output_count, scratch->keys.size())
            : node::accel::detail::ExecuteLinearScatter(
                  *scratch,
                  reinterpret_cast<const kernel::u64 *>(context.port(0u).data),
                  reinterpret_cast<const kernel::u32 *>(context.port(1u).data),
                  reinterpret_cast<kernel::u64 *>(context.port(2u).data),
                  plan.element_count, plan.output_count, scratch->keys.size());
    return finish_cpu_primitive(result.ok, result.reason);
  }

  return finish_cpu_primitive(false, "compute_primitive_unsupported");
}

} // namespace rund::compute::detail
