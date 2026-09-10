#include "internal.hpp"

#include "../../type.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail {

ExprRef binary(const ExprOp operation, ExprRef left, ExprRef right) {
  if (!expression_build::valid(left) || !expression_build::valid(right) ||
      left.state != right.state || left.type != right.type) {
    if (left.state != nullptr) {
      expression_build::set_error(
          left.state, Status::fail(Reason::ExpressionContextMismatch));
    }
    return ExprRef{std::move(left.state), 0, left.type, left.fixed_format};
  }
  FixedFormat format = left.fixed_format;
  if (type_fixed(left.type)) {
    if (left.fixed_format.integer_bits == 0u ||
        right.fixed_format.integer_bits == 0u ||
        left.fixed_format.rounding != right.fixed_format.rounding ||
        left.fixed_format.overflow != right.fixed_format.overflow) {
      expression_build::set_error(left.state,
                                  Status::fail(Reason::FixedFormatMismatch));
      return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
    }
    if (expression_build::stored_binary(operation) &&
        (!expression_build::stored_format(left.type, left.fixed_format) ||
         !expression_build::stored_format(right.type, right.fixed_format) ||
         left.fixed_format.integer_bits != right.fixed_format.integer_bits ||
         left.fixed_format.fraction_bits != right.fixed_format.fraction_bits)) {
      expression_build::set_error(left.state,
                                  Status::fail(Reason::FixedQuantizeRequired));
      return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
    }
    const unsigned left_integer = left.fixed_format.integer_bits;
    const unsigned left_fraction = left.fixed_format.fraction_bits;
    const unsigned right_integer = right.fixed_format.integer_bits;
    const unsigned right_fraction = right.fixed_format.fraction_bits;
    format.approximation =
        static_cast<unsigned>(left.fixed_format.approximation) >=
                static_cast<unsigned>(right.fixed_format.approximation)
            ? left.fixed_format.approximation
            : right.fixed_format.approximation;
    if (operation == ExprOp::Multiply) {
      if (left_integer + right_integer + left_fraction + right_fraction >
          128u) {
        expression_build::set_error(
            left.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
      }
      format.integer_bits =
          static_cast<unsigned char>(left_integer + right_integer);
      format.fraction_bits =
          static_cast<unsigned char>(left_fraction + right_fraction);
    } else if (operation == ExprOp::Add || operation == ExprOp::Subtract) {
      const unsigned integer = std::max(left_integer, right_integer) + 1u;
      const unsigned fraction = std::max(left_fraction, right_fraction);
      if (integer + fraction > 128u) {
        expression_build::set_error(
            left.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    } else {
      const unsigned integer = std::max(left_integer, right_integer);
      const unsigned fraction = std::max(left_fraction, right_fraction);
      if (integer + fraction > 128u) {
        expression_build::set_error(
            left.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    }
    if (expression_build::approximate_binary(operation)) {
      format.approximation = Approximation::Deterministic;
    }
  }
  return expression_build::append(left.state, ExprNode{
                                                  .operation = operation,
                                                  .type = left.type,
                                                  .fixed_format = format,
                                                  .left = left.node,
                                                  .right = right.node,
                                              });
}

ExprRef ternary(const ExprOp operation, ExprRef first, ExprRef second,
                ExprRef third) {
  if (!expression_build::valid(first) || !expression_build::valid(second) ||
      !expression_build::valid(third) || first.state != second.state ||
      first.state != third.state || second.type != third.type) {
    if (first.state != nullptr) {
      expression_build::set_error(
          first.state, Status::fail(Reason::ExpressionContextMismatch));
    }
    return ExprRef{std::move(first.state), 0, second.type, second.fixed_format};
  }
  FixedFormat format = second.fixed_format;
  if (type_fixed(second.type)) {
    const auto same_policy = [](const FixedFormat left,
                                const FixedFormat right) noexcept {
      return left.rounding == right.rounding && left.overflow == right.overflow;
    };
    if (second.fixed_format.integer_bits == 0u ||
        third.fixed_format.integer_bits == 0u ||
        !same_policy(second.fixed_format, third.fixed_format) ||
        (operation != ExprOp::Select &&
         (first.fixed_format.integer_bits == 0u ||
          !same_policy(first.fixed_format, second.fixed_format)))) {
      expression_build::set_error(first.state,
                                  Status::fail(Reason::FixedFormatMismatch));
      return ExprRef{std::move(first.state), 0u, second.type,
                     second.fixed_format};
    }
    format.approximation =
        static_cast<unsigned>(second.fixed_format.approximation) >=
                static_cast<unsigned>(third.fixed_format.approximation)
            ? second.fixed_format.approximation
            : third.fixed_format.approximation;
    if (operation != ExprOp::Select &&
        static_cast<unsigned>(first.fixed_format.approximation) >
            static_cast<unsigned>(format.approximation)) {
      format.approximation = first.fixed_format.approximation;
    }
    if (operation == ExprOp::MulAddFixed) {
      const unsigned product_integer =
          first.fixed_format.integer_bits + second.fixed_format.integer_bits;
      const unsigned product_fraction =
          first.fixed_format.fraction_bits + second.fixed_format.fraction_bits;
      const unsigned addend_integer = third.fixed_format.integer_bits;
      const unsigned integer = product_integer > addend_integer
                                   ? product_integer
                                   : addend_integer + 1u;
      const unsigned fraction =
          std::max(product_fraction,
                   static_cast<unsigned>(third.fixed_format.fraction_bits));
      if (integer + fraction > 128u) {
        expression_build::set_error(
            first.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(first.state), 0u, second.type,
                       second.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    } else {
      unsigned integer =
          std::max(static_cast<unsigned>(second.fixed_format.integer_bits),
                   static_cast<unsigned>(third.fixed_format.integer_bits));
      unsigned fraction =
          std::max(static_cast<unsigned>(second.fixed_format.fraction_bits),
                   static_cast<unsigned>(third.fixed_format.fraction_bits));
      if (operation != ExprOp::Select) {
        integer = std::max(
            integer, static_cast<unsigned>(first.fixed_format.integer_bits));
        fraction = std::max(
            fraction, static_cast<unsigned>(first.fixed_format.fraction_bits));
      }
      if (integer + fraction > 128u) {
        expression_build::set_error(
            first.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(first.state), 0u, second.type,
                       second.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    }
  }
  return expression_build::append(first.state, ExprNode{
                                                   .operation = operation,
                                                   .type = second.type,
                                                   .fixed_format = format,
                                                   .left = first.node,
                                                   .right = second.node,
                                                   .third = third.node,
                                               });
}

} // namespace rund::compute::detail
