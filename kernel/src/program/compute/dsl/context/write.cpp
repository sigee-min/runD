#include <kernel/program/compute/dsl/expression/context.hpp>

#include <limits>

namespace rund::compute_dsl::detail {

void BuildContext::write_node(const rund::kernel::u32 binding,
                              rund::kernel::u32 value) noexcept {
  if (!valid_binding(binding, BindingKind::Write)) {
    reject("compute_binding_invalid");
    return;
  }
  if (!valid_node(value)) {
    reject("compute_value_invalid");
    return;
  }
  const auto source_domain = node_domain(value);
  const auto target_domain = binding_domain(binding);
  const bool target_unsigned =
      target_domain == rund::kernel::ComputeDomain::U32 ||
      target_domain == rund::kernel::ComputeDomain::U64;
  const bool canonical_unsigned_mask =
      target_unsigned && canonical_mask_source(value);
  const ScalarMode target_mode = (*bindings_)[binding].numeric_mode;
  const bool target_fixed = target_domain == rund::kernel::ComputeDomain::Fixed;
  const bool target_matches_graph =
      scalar_mode_ == ScalarMode::Unspecified ? true
      : fixed_mode() ? target_fixed && target_mode == scalar_mode_
                     : !target_fixed;
  if ((!target_matches_graph && !(fixed_mode() && canonical_unsigned_mask)) ||
      (source_domain != target_domain && !canonical_unsigned_mask)) {
    reject("compute_value_invalid");
    return;
  }
  if (scalar_mode_ == ScalarMode::FixedLane32 ||
      scalar_mode_ == ScalarMode::FixedLane64) {
    const rund::kernel::ComputeIrNode &source = nodes_[value - 1u];
    if (source.op != rund::kernel::IrOp::Quantize) {
      value = storage_quantize_node(value);
      if (!valid_node(value)) {
        reject("compute_quantize_invalid");
        return;
      }
    }
  }
  const rund::kernel::ComputeFixedFormat write_format =
      fixed_mode() ? source_format(value) : rund::kernel::ComputeFixedFormat{};
  (void)append_node(rund::kernel::IrOp::Write, value, 0u, binding, write_format,
                    source_domain);
  ++write_count_;
}

void BuildContext::write_checked_ordinal_node(
    const rund::kernel::u32 binding, const rund::kernel::u32 value) noexcept {
  if (!valid_binding(binding, BindingKind::Write)) {
    reject("compute_binding_invalid");
    return;
  }
  if (!valid_node(value)) {
    reject("compute_value_invalid");
    return;
  }
  const auto target_domain = binding_domain(binding);
  const auto source_domain =
      checked_ordinal_source_domain(value, target_domain);
  const auto integer_domain = [](const rund::kernel::ComputeDomain domain) {
    return domain == rund::kernel::ComputeDomain::I32 ||
           domain == rund::kernel::ComputeDomain::U32 ||
           domain == rund::kernel::ComputeDomain::I64 ||
           domain == rund::kernel::ComputeDomain::U64;
  };
  const bool same_width = (source_domain == rund::kernel::ComputeDomain::I32 ||
                           source_domain == rund::kernel::ComputeDomain::U32) ==
                          (target_domain == rund::kernel::ComputeDomain::I32 ||
                           target_domain == rund::kernel::ComputeDomain::U32);
  if (!integer_domain(source_domain) || !integer_domain(target_domain) ||
      !same_width || source_domain == target_domain) {
    reject("compute_value_invalid");
    return;
  }
  (void)append_node(
      rund::kernel::IrOp::Write, value,
      static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::CheckedOrdinal),
      binding, {}, target_domain);
  ++write_count_;
}

void BuildContext::write_boundary_mask_node(
    const rund::kernel::u32 binding, const rund::kernel::u32 value,
    const rund::kernel::ComputeFixedFormat target_format) noexcept {
  if (!valid_binding(binding, BindingKind::Write)) {
    reject("compute_binding_invalid");
    return;
  }
  if (!valid_node(value)) {
    reject("compute_value_invalid");
    return;
  }
  const auto target_domain = binding_domain(binding);
  const auto source_domain = boundary_mask_source_domain(value);
  if (target_domain == rund::kernel::ComputeDomain::U32 ||
      target_domain == rund::kernel::ComputeDomain::U64) {
    if (!rund::kernel::ComputeFixedFormatAbsent(target_format)) {
      reject("compute_value_invalid");
      return;
    }
    write_node(binding, value);
    return;
  }
  const bool source_integer =
      source_domain == rund::kernel::ComputeDomain::I32 ||
      source_domain == rund::kernel::ComputeDomain::U32 ||
      source_domain == rund::kernel::ComputeDomain::I64 ||
      source_domain == rund::kernel::ComputeDomain::U64;
  const bool target_supported =
      target_domain == rund::kernel::ComputeDomain::I32 ||
      target_domain == rund::kernel::ComputeDomain::I64 ||
      target_domain == rund::kernel::ComputeDomain::Fixed;
  const ScalarMode target_mode = (*bindings_)[binding].numeric_mode;
  const bool same_width = WideMode(target_mode) == WideMode(scalar_mode_);
  const rund::kernel::ComputeScalar scalar =
      WideMode(target_mode) ? rund::kernel::ComputeScalar::Lane64
                            : rund::kernel::ComputeScalar::Lane32;
  const bool format_valid =
      target_domain == rund::kernel::ComputeDomain::Fixed
          ? rund::kernel::ComputeFixedFormatValid(scalar, target_format)
          : rund::kernel::ComputeFixedFormatAbsent(target_format);
  if (!source_integer || !target_supported || !same_width || !format_valid ||
      source_domain == target_domain) {
    reject("compute_value_invalid");
    return;
  }
  (void)append_node(
      rund::kernel::IrOp::Write, value,
      static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::BoundaryMask),
      binding, target_format, target_domain);
  ++write_count_;
}

bool BuildContext::canonical_mask_source(
    rund::kernel::u32 source) const noexcept {
  if (!valid_node(source)) {
    return false;
  }
  if (nodes_[source - 1u].op == rund::kernel::IrOp::Quantize) {
    source = nodes_[source - 1u].lhs;
    if (!valid_node(source)) {
      return false;
    }
  }
  const auto &select = nodes_[source - 1u];
  if (select.op != rund::kernel::IrOp::Select || !valid_node(select.rhs) ||
      !valid_node(select.aux)) {
    return false;
  }
  const auto &one = nodes_[select.rhs - 1u];
  const auto &zero = nodes_[select.aux - 1u];
  return one.op == rund::kernel::IrOp::Constant && one.lhs == 1u &&
         one.rhs == 0u && one.aux == 0u &&
         zero.op == rund::kernel::IrOp::Constant && zero.lhs == 0u &&
         zero.rhs == 0u && zero.aux == 0u;
}

bool BuildContext::canonical_zero(
    const rund::kernel::u32 source) const noexcept {
  return valid_node(source) &&
         nodes_[source - 1u].op == rund::kernel::IrOp::Constant &&
         nodes_[source - 1u].lhs == 0u && nodes_[source - 1u].rhs == 0u &&
         nodes_[source - 1u].aux == 0u;
}

rund::kernel::ComputeDomain BuildContext::checked_ordinal_source_domain(
    const rund::kernel::u32 source,
    const rund::kernel::ComputeDomain target_domain) const noexcept {
  const auto unknown = static_cast<rund::kernel::ComputeDomain>(0u);
  if (!valid_node(source)) {
    return unknown;
  }
  const auto &select = nodes_[source - 1u];
  if (select.op != rund::kernel::IrOp::Select || !valid_node(select.lhs) ||
      !valid_node(select.rhs) || !canonical_zero(select.aux)) {
    return unknown;
  }
  const auto &predicate = nodes_[select.lhs - 1u];
  const auto value_domain = node_domain(select.rhs);
  const bool signed_to_unsigned =
      (value_domain == rund::kernel::ComputeDomain::I32 &&
       target_domain == rund::kernel::ComputeDomain::U32) ||
      (value_domain == rund::kernel::ComputeDomain::I64 &&
       target_domain == rund::kernel::ComputeDomain::U64);
  const bool unsigned_to_signed =
      (value_domain == rund::kernel::ComputeDomain::U32 &&
       target_domain == rund::kernel::ComputeDomain::I32) ||
      (value_domain == rund::kernel::ComputeDomain::U64 &&
       target_domain == rund::kernel::ComputeDomain::I64);
  if (predicate.lhs != select.rhs) {
    return unknown;
  }
  if (signed_to_unsigned) {
    return predicate.op == rund::kernel::IrOp::Ge && predicate.rhs == select.aux
               ? value_domain
               : unknown;
  }
  if (!unsigned_to_signed || !valid_node(predicate.rhs)) {
    return unknown;
  }
  const auto &limit = nodes_[predicate.rhs - 1u];
  const rund::kernel::u64 expected =
      value_domain == rund::kernel::ComputeDomain::U64
          ? static_cast<rund::kernel::u64>(
                std::numeric_limits<rund::kernel::i64>::max())
          : static_cast<rund::kernel::u64>(
                std::numeric_limits<rund::kernel::i32>::max());
  const rund::kernel::u64 bits =
      static_cast<rund::kernel::u64>(limit.lhs) |
      (static_cast<rund::kernel::u64>(limit.rhs) << 32u);
  return predicate.op == rund::kernel::IrOp::LeUnsigned &&
                 limit.op == rund::kernel::IrOp::Constant && limit.aux == 0u &&
                 bits == expected
             ? value_domain
             : unknown;
}

rund::kernel::ComputeDomain BuildContext::boundary_mask_source_domain(
    const rund::kernel::u32 source) const noexcept {
  const auto unknown = static_cast<rund::kernel::ComputeDomain>(0u);
  if (!valid_node(source)) {
    return unknown;
  }
  const auto &select = nodes_[source - 1u];
  if (select.op != rund::kernel::IrOp::Select || !valid_node(select.lhs) ||
      !valid_node(select.rhs) || !valid_node(select.aux)) {
    return unknown;
  }
  const auto &one = nodes_[select.rhs - 1u];
  const auto &predicate = nodes_[select.lhs - 1u];
  if (one.op != rund::kernel::IrOp::Constant || one.lhs != 1u ||
      one.rhs != 0u || one.aux != 0u || !canonical_zero(select.aux) ||
      predicate.op != rund::kernel::IrOp::Ne || !valid_node(predicate.lhs) ||
      predicate.rhs != select.aux) {
    return unknown;
  }
  return node_domain(predicate.lhs);
}

} // namespace rund::compute_dsl::detail
