#include "internal.hpp"

#include "../../type.hpp"

#include <utility>

namespace rund::compute::detail {

ExprRef retype_expr(ExprRef value, const Type type) {
  if (!expression_build::valid(value) || !valid_type(value.type) ||
      !valid_type(type) || type_bytes(value.type) != type_bytes(type)) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, value.fixed_format};
  }
  value.type = type;
  return value;
}

ExprRef checked_ordinal_expr(ExprRef value, const Type type) {
  if (!expression_build::valid(value) || !valid_type(value.type) ||
      !valid_type(type) || type_fixed(value.type) || type_fixed(type) ||
      type_bytes(value.type) != type_bytes(type)) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, {}};
  }
  return expression_build::append(value.state,
                                  ExprNode{
                                      .operation = ExprOp::CheckedOrdinal,
                                      .type = type,
                                      .left = value.node,
                                  });
}

ExprRef boundary_mask_expr(ExprRef value, const Type type,
                           const FixedFormat fixed_format) {
  const bool source_supported = valid_type(value.type);
  const bool target_supported = valid_type(type);
  const bool source_integer = source_supported && !type_fixed(value.type);
  const bool format_valid =
      type_fixed(type) ? expression_build::stored_format(type, fixed_format)
                       : fixed_format == FixedFormat{};
  if (!expression_build::valid(value) || !source_supported || !source_integer ||
      !target_supported || type_bytes(value.type) != type_bytes(type) ||
      !format_valid) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, fixed_format};
  }
  return expression_build::append(value.state,
                                  ExprNode{
                                      .operation = ExprOp::BoundaryMask,
                                      .type = type,
                                      .fixed_format = fixed_format,
                                      .left = value.node,
                                  });
}

ExprRef with_fixed_format(ExprRef value, const FixedFormat fixed_format) {
  if (!expression_build::valid(value) || !type_fixed(value.type)) {
    return value;
  }
  if (value.fixed_format.integer_bits != 0u) {
    return value;
  }
  value.fixed_format = fixed_format;
  value.state->nodes[value.node - 1u].fixed_format = fixed_format;
  return value;
}

ExprRef quantize_expr(ExprRef value, const Type target,
                      const FixedFormat fixed_format) {
  if (!expression_build::valid(value) && value.state != nullptr &&
      !value.state->status) {
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  if (!expression_build::valid(value) || !type_fixed(value.type) ||
      !type_fixed(target) || fixed_format.integer_bits == 0u ||
      fixed_format.fraction_bits == 0u ||
      static_cast<unsigned>(fixed_format.integer_bits) +
              fixed_format.fraction_bits !=
          type_bytes(target) * 8u) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::QuantizeFormatInvalid));
    }
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  if (value.fixed_format.approximation == Approximation::Deterministic &&
      fixed_format.approximation != Approximation::Deterministic) {
    expression_build::set_error(
        value.state, Status::fail(Reason::FixedApproximationDowngrade));
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  const ExprNode &source = value.state->nodes[value.node - 1u];
  if (source.operation == ExprOp::Quantize && value.type == target &&
      value.fixed_format == fixed_format) {
    return value;
  }
  return expression_build::append(value.state,
                                  ExprNode{
                                      .operation = ExprOp::Quantize,
                                      .type = target,
                                      .fixed_format = fixed_format,
                                      .left = value.node,
                                  });
}

ExprRef make_mask(ExprRef predicate) {
  return make_mask(std::move(predicate), Type::U32);
}

ExprRef make_mask(ExprRef predicate, const Type output) {
  if (!expression_build::valid(predicate)) {
    if (predicate.state != nullptr) {
      expression_build::set_error(predicate.state,
                                  Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(predicate.state), 0, output};
  }
  if (output != Type::U32 && output != Type::U64) {
    expression_build::set_error(predicate.state,
                                Status::fail(Reason::ExpressionTypeMismatch));
    return ExprRef{std::move(predicate.state), 0, output};
  }
  return expression_build::append(predicate.state,
                                  ExprNode{
                                      .operation = ExprOp::Mask,
                                      .type = output,
                                      .fixed_format = predicate.fixed_format,
                                      .left = predicate.node,
                                  });
}

bool is_width_mask(const ExprRef &expression, const Type input) noexcept {
  return expression.state != nullptr && expression.node != 0u &&
         expression.node <= expression.state->nodes.size() &&
         (expression.type == Type::U32 || expression.type == Type::U64) &&
         (type_bytes(input) == sizeof(std::uint32_t) ||
          type_bytes(input) == sizeof(std::uint64_t)) &&
         expression.state->nodes[expression.node - 1u].operation ==
             ExprOp::Mask;
}

} // namespace rund::compute::detail
