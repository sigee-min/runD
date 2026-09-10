#include "internal.hpp"

#include "../../type.hpp"

#include <utility>

namespace rund::compute::detail {

ExprRef unary(const ExprOp operation, ExprRef value) {
  if (!expression_build::valid(value)) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (type_fixed(value.type) && expression_build::stored_unary(operation) &&
      !expression_build::stored_format(value.type, value.fixed_format)) {
    expression_build::set_error(value.state,
                                Status::fail(Reason::FixedQuantizeRequired));
    return ExprRef{std::move(value.state), 0u, value.type, value.fixed_format};
  }
  FixedFormat format = value.fixed_format;
  if (type_fixed(value.type) &&
      expression_build::approximate_unary(operation)) {
    format.approximation = Approximation::Deterministic;
  }
  if (type_fixed(value.type) &&
      (operation == ExprOp::Negate || operation == ExprOp::Abs ||
       operation == ExprOp::AbsMagnitude)) {
    const unsigned width =
        static_cast<unsigned>(format.integer_bits) + format.fraction_bits;
    if (width >= 128u) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::FixedPrecisionCapacity));
      return ExprRef{std::move(value.state), 0u, value.type,
                     value.fixed_format};
    }
    ++format.integer_bits;
  }
  return expression_build::append(value.state, ExprNode{
                                                   .operation = operation,
                                                   .type = value.type,
                                                   .fixed_format = format,
                                                   .left = value.node,
                                               });
}

ExprRef shift(const ExprOp operation, ExprRef value,
              const std::uint32_t amount) {
  if (!expression_build::valid(value)) {
    if (value.state != nullptr) {
      expression_build::set_error(value.state,
                                  Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (!valid_type(value.type)) {
    expression_build::set_error(value.state,
                                Status::fail(Reason::ExpressionTypeMismatch));
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (amount >= type_bytes(value.type) * 8u) {
    expression_build::set_error(value.state,
                                Status::fail(Reason::ShiftCountInvalid));
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (type_fixed(value.type) &&
      !expression_build::stored_format(value.type, value.fixed_format)) {
    expression_build::set_error(value.state,
                                Status::fail(Reason::FixedQuantizeRequired));
    return ExprRef{std::move(value.state), 0u, value.type, value.fixed_format};
  }
  return expression_build::append(value.state,
                                  ExprNode{.operation = operation,
                                           .type = value.type,
                                           .fixed_format = value.fixed_format,
                                           .left = value.node,
                                           .immediate = amount});
}

} // namespace rund::compute::detail
