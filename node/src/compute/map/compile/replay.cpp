#include "model.hpp"

#include "../../type.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace rund::compute::detail {
namespace {

[[nodiscard]] KernelExpr emit_unary(const kernel::IrOp operation,
                                    const KernelExpr value,
                                    const ExprNode &node) {
  return type_fixed(node.type)
             ? compute_dsl::detail::UnaryFormatted(
                   operation, value, kernel_format(node.fixed_format))
             : compute_dsl::detail::Unary(operation, value);
}

[[nodiscard]] KernelExpr emit_binary(const kernel::IrOp operation,
                                     const KernelExpr left,
                                     const KernelExpr right,
                                     const ExprNode &node) {
  if (operation == kernel::IrOp::MulWrap) {
    return compute_dsl::detail::Binary(operation, left, right);
  }
  return type_fixed(node.type)
             ? compute_dsl::detail::BinaryFormatted(
                   operation, left, right, kernel_format(node.fixed_format))
             : compute_dsl::detail::Binary(operation, left, right);
}

[[nodiscard]] KernelExpr emit_ternary(const kernel::IrOp operation,
                                      const KernelExpr first,
                                      const KernelExpr second,
                                      const KernelExpr third,
                                      const ExprNode &node) {
  return type_fixed(node.type)
             ? compute_dsl::detail::TernaryFormatted(
                   operation, first, second, third,
                   kernel_format(node.fixed_format))
             : compute_dsl::detail::Ternary(operation, first, second, third);
}

} // namespace

std::optional<KernelExpr>
replay(const ExprRef &expression, const std::span<const KernelExpr> inputs,
       const KernelExpr constant_anchor, const KernelExpr logical_index,
       std::vector<KernelExpr> &values, std::vector<std::uint8_t> &state) {
  if (expression.state == nullptr || expression.node == 0u ||
      expression.node > expression.state->nodes.size()) {
    return std::nullopt;
  }
  try {
    if (values.size() < expression.node) {
      values.resize(expression.node);
    }
    if (state.size() < expression.node) {
      state.resize(expression.node, 0u);
    }
    const auto visit = [&](auto &&self, const std::uint32_t ref) -> bool {
      if (ref == 0u || ref > expression.node) {
        return false;
      }
      const std::size_t index = ref - 1u;
      if (state[index] == 2u) {
        return true;
      }
      if (state[index] == 1u) {
        return false;
      }
      state[index] = 1u;
      const ExprNode &node = expression.state->nodes[index];
      const auto operand = [&](const std::uint32_t value) {
        return self(self, value);
      };
      if (node.operation == ExprOp::Input) {
        if (node.left >= inputs.size()) {
          return false;
        }
      } else {
        const std::uint8_t arity = expr_arity(node.operation);
        if (arity == InvalidArity || (arity >= 1u && !operand(node.left)) ||
            (arity >= 2u && !operand(node.right)) ||
            (arity == 3u && !operand(node.third))) {
          return false;
        }
      }

      switch (node.operation) {
      case ExprOp::Input:
        values[index] = inputs[node.left];
        break;
      case ExprOp::Constant:
        values[index] =
            type_fixed(node.type)
                ? compute_dsl::detail::FormattedConstant(
                      constant_anchor, node.bits,
                      kernel_format(node.fixed_format))
                : compute_dsl::detail::TypedConstant(
                      constant_anchor, numeric_mode(node.type), node.bits);
        break;
      case ExprOp::Index:
        values[index] = type_fixed(node.type)
                            ? logical_index
                            : compute_dsl::detail::TypedIndex(
                                  logical_index, numeric_mode(node.type));
        break;
      case ExprOp::Negate:
      case ExprOp::Abs:
      case ExprOp::AbsMagnitude:
      case ExprOp::Sign:
      case ExprOp::BitNot:
      case ExprOp::NegPositiveFixed:
      case ExprOp::Reciprocal:
      case ExprOp::Sqrt:
      case ExprOp::Rsqrt:
      case ExprOp::Sin:
      case ExprOp::Cos:
      case ExprOp::Tan:
      case ExprOp::Exp:
      case ExprOp::Log: {
        const auto operation = unary_op(node.operation);
        if (!operation) {
          return false;
        }
        values[index] = emit_unary(*operation, values[node.left - 1], node);
        break;
      }
      case ExprOp::Quantize:
        values[index] =
            emit_unary(kernel::IrOp::Quantize, values[node.left - 1], node);
        break;
      case ExprOp::CheckedOrdinal:
        if (const kernel::ComputeDomain source_domain =
                type_domain(expression.state->nodes[node.left - 1u].type);
            source_domain == kernel::ComputeDomain::I32 ||
            source_domain == kernel::ComputeDomain::I64) {
          const KernelExpr zero = compute_dsl::detail::TypedConstant(
              values[node.left - 1u],
              numeric_mode(expression.state->nodes[node.left - 1u].type), 0u);
          const KernelExpr representable = compute_dsl::detail::Binary(
              kernel::IrOp::Ge, values[node.left - 1u], zero);
          values[index] =
              compute_dsl::detail::Ternary(kernel::IrOp::Select, representable,
                                           values[node.left - 1u], zero);
        } else {
          const std::uint64_t maximum =
              type_bytes(node.type) == 8u
                  ? static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())
                  : static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max());
          const KernelExpr limit = compute_dsl::detail::TypedConstant(
              values[node.left - 1u],
              numeric_mode(expression.state->nodes[node.left - 1u].type),
              maximum);
          const KernelExpr zero = compute_dsl::detail::TypedConstant(
              values[node.left - 1u],
              numeric_mode(expression.state->nodes[node.left - 1u].type), 0u);
          const KernelExpr representable = compute_dsl::detail::Binary(
              kernel::IrOp::LeUnsigned, values[node.left - 1u], limit);
          values[index] =
              compute_dsl::detail::Ternary(kernel::IrOp::Select, representable,
                                           values[node.left - 1u], zero);
        }
        break;
      case ExprOp::BoundaryMask: {
        const KernelExpr zero = compute_dsl::detail::TypedConstant(
            values[node.left - 1u],
            numeric_mode(expression.state->nodes[node.left - 1u].type), 0u);
        const KernelExpr predicate = compute_dsl::detail::Binary(
            kernel::IrOp::Ne, values[node.left - 1u], zero);
        const KernelExpr one = compute_dsl::detail::TypedConstant(
            values[node.left - 1u],
            numeric_mode(expression.state->nodes[node.left - 1u].type), 1u);
        values[index] = compute_dsl::detail::Ternary(kernel::IrOp::Select,
                                                     predicate, one, zero);
        break;
      }
      case ExprOp::ShiftLeft:
        values[index] = compute_dsl::detail::ConstShift(
            kernel::IrOp::ShlConst, values[node.left - 1], node.immediate);
        break;
      case ExprOp::ShiftRightLogical:
        values[index] = compute_dsl::detail::ConstShift(
            kernel::IrOp::ShrLogicalConst, values[node.left - 1],
            node.immediate);
        break;
      case ExprOp::ShiftRightArithmetic:
        values[index] = compute_dsl::detail::ConstShift(
            kernel::IrOp::ShrArithmeticConst, values[node.left - 1],
            node.immediate);
        break;
      case ExprOp::PredicateNot:
        values[index] = compute_dsl::detail::Unary(kernel::IrOp::PredicateNot,
                                                   values[node.left - 1]);
        break;
      case ExprOp::Mask: {
        const KernelExpr one =
            compute_dsl::detail::StorageConstant(values[node.left - 1], 1u);
        const KernelExpr zero =
            compute_dsl::detail::StorageConstant(values[node.left - 1], 0u);
        values[index] = compute_dsl::detail::Ternary(
            kernel::IrOp::Select, values[node.left - 1], one, zero);
        if (node.fixed_format.integer_bits != 0u) {
          values[index] = compute_dsl::detail::StorageQuantize(values[index]);
        }
        break;
      }
      case ExprOp::Add:
      case ExprOp::Subtract:
      case ExprOp::Multiply:
      case ExprOp::MultiplyWrap:
      case ExprOp::Divide:
      case ExprOp::BitAnd:
      case ExprOp::BitOr:
      case ExprOp::BitXor:
      case ExprOp::Min:
      case ExprOp::Max:
      case ExprOp::Equal:
      case ExprOp::NotEqual:
      case ExprOp::Less:
      case ExprOp::LessEqual:
      case ExprOp::Greater:
      case ExprOp::GreaterEqual:
      case ExprOp::PredicateAnd:
      case ExprOp::PredicateOr:
      case ExprOp::AddSat:
      case ExprOp::AddSatUnsigned:
      case ExprOp::SubSat:
      case ExprOp::MulFixed:
      case ExprOp::MulFixedScaled:
      case ExprOp::MulUnsignedFixed:
      case ExprOp::Atan2: {
        const auto operation = binary_op(node.operation, node.type);
        if (!operation) {
          return false;
        }
        values[index] = emit_binary(*operation, values[node.left - 1],
                                    values[node.right - 1], node);
        break;
      }
      case ExprOp::Clamp:
        values[index] =
            emit_ternary(kernel::CanonicalIrOpForDomain(kernel::IrOp::Clamp,
                                                        type_domain(node.type)),
                         values[node.left - 1], values[node.right - 1],
                         values[node.third - 1], node);
        break;
      case ExprOp::Select:
        values[index] =
            emit_ternary(kernel::IrOp::Select, values[node.left - 1],
                         values[node.right - 1], values[node.third - 1], node);
        break;
      case ExprOp::MulAddFixed:
        values[index] =
            emit_ternary(kernel::IrOp::MulAddFixed, values[node.left - 1],
                         values[node.right - 1], values[node.third - 1], node);
        break;
      default:
        return false;
      }
      state[index] = 2u;
      return true;
    };
    if (!visit(visit, expression.node)) {
      return std::nullopt;
    }
    return values[expression.node - 1u];
  } catch (const std::bad_alloc &) {
    return std::nullopt;
  }
}

} // namespace rund::compute::detail
