#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

rund::kernel::u32
BuildContext::param_node(const std::string_view name) noexcept {
  const rund::kernel::u32 binding = binding_index(name, BindingKind::Param);
  if (!valid_binding(binding, BindingKind::Param)) {
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Param, 0u, 0u, binding, value_format(),
                     binding_domain(binding));
}

rund::kernel::u32
BuildContext::read_node(const rund::kernel::u32 binding) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Read, 0u, 0u, binding, value_format(),
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_node(
    const rund::kernel::u32 binding,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Read, 0u, 0u, binding, format,
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_uniform_node(
    const rund::kernel::u32 binding,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::ReadUniform, 0u, 0u, binding, format,
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_at_node(
    const rund::kernel::u32 binding, const rund::kernel::u32 index,
    const rund::kernel::u32 count,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read) ||
      !valid_binding(index, BindingKind::Read) || binding == index ||
      count == 0u || !binding_value_mode_valid(binding) ||
      bindings_ == nullptr ||
      (*bindings_)[index].numeric_mode != ScalarMode::U32 ||
      (*bindings_)[index].element_bytes != sizeof(rund::kernel::u32)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::ReadAt, index, count, binding, format,
                     binding_domain(binding));
}

} // namespace rund::compute_dsl::detail
