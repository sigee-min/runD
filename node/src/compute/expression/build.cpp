#include "state.hpp"

#include "../../hash/fnv.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace rund::compute::detail {
namespace {

constexpr std::size_t MinCanonicalSlots = 8u;

[[nodiscard]] constexpr std::uint64_t node_hash(const ExprNode &node) noexcept {
  ::rund::node::hash_detail::Fnv hash{};
  hash.Byte(static_cast<std::uint8_t>(node.operation));
  hash.Byte(static_cast<std::uint8_t>(node.type));
  hash.Byte(node.fixed_format.integer_bits);
  hash.Byte(node.fixed_format.fraction_bits);
  hash.Byte(static_cast<std::uint8_t>(node.fixed_format.rounding));
  hash.Byte(static_cast<std::uint8_t>(node.fixed_format.overflow));
  hash.Byte(static_cast<std::uint8_t>(node.fixed_format.approximation));
  hash.Number(node.left);
  hash.Number(node.right);
  hash.Number(node.third);
  hash.Number(node.immediate);
  hash.Number(node.bits);
  return hash.Finish();
}

[[nodiscard]] std::size_t canonical_slot(const ExprState &state,
                                         const ExprNode &node) noexcept {
  return static_cast<std::size_t>(node_hash(node)) &
         (state.canonical_slots.size() - 1u);
}

[[nodiscard]] std::uint32_t find_canonical(const ExprState &state,
                                           const ExprNode &node) noexcept {
  if (state.canonical_slots.empty()) {
    return 0u;
  }
  const std::size_t mask = state.canonical_slots.size() - 1u;
  std::size_t slot = canonical_slot(state, node);
  for (std::size_t probe = 0u; probe < state.canonical_slots.size(); ++probe) {
    const std::uint16_t stored = state.canonical_slots[slot];
    if (stored == 0u) {
      return 0u;
    }
    const std::size_t index = static_cast<std::size_t>(stored - 1u);
    if (index < state.nodes.size() && state.nodes[index] == node) {
      return stored;
    }
    slot = (slot + 1u) & mask;
  }
  return 0u;
}

void insert_canonical(ExprState &state, const std::size_t index) noexcept {
  const ExprNode &node = state.nodes[index];
  const std::size_t mask = state.canonical_slots.size() - 1u;
  std::size_t slot = canonical_slot(state, node);
  for (std::size_t probe = 0u; probe < state.canonical_slots.size(); ++probe) {
    std::uint16_t &stored = state.canonical_slots[slot];
    if (stored == 0u) {
      stored = static_cast<std::uint16_t>(index + 1u);
      return;
    }
    const std::size_t existing = static_cast<std::size_t>(stored - 1u);
    if (existing < state.nodes.size() && state.nodes[existing] == node) {
      return;
    }
    slot = (slot + 1u) & mask;
  }
}

void prepare_canonical(ExprState &state, const std::size_t required) {
  const bool synchronized = state.canonical_nodes == state.nodes.size();
  const bool has_capacity = !state.canonical_slots.empty() &&
                            required <= state.canonical_slots.size() / 2u;
  if (synchronized && (required == 0u || has_capacity)) {
    return;
  }
  std::size_t slots = MinCanonicalSlots;
  while (required > slots / 2u) {
    slots *= 2u;
  }
  std::vector<std::uint16_t> rebuilt(slots, 0u);
  state.canonical_slots.swap(rebuilt);
  for (std::size_t index = 0u; index < state.nodes.size(); ++index) {
    insert_canonical(state, index);
  }
  state.canonical_nodes = state.nodes.size();
}

void set_error(const std::shared_ptr<ExprState> &state, Status status) {
  if (state != nullptr && state->status) {
    state->status = std::move(status);
  }
}

[[nodiscard]] constexpr bool fixed(const Type type) noexcept {
  return type == Type::FixedLane32 || type == Type::FixedLane64;
}

[[nodiscard]] constexpr std::size_t bytes(const Type type) noexcept {
  return type == Type::I64 || type == Type::U64 || type == Type::FixedLane64
             ? 8u
             : 4u;
}

[[nodiscard]] constexpr bool stored_unary(const ExprOp operation) noexcept {
  return operation == ExprOp::NegPositiveFixed || operation == ExprOp::BitNot ||
         operation == ExprOp::Reciprocal || operation == ExprOp::Sqrt ||
         operation == ExprOp::Rsqrt || operation == ExprOp::Sin ||
         operation == ExprOp::Cos || operation == ExprOp::Tan ||
         operation == ExprOp::Exp || operation == ExprOp::Log;
}

[[nodiscard]] constexpr bool stored_binary(const ExprOp operation) noexcept {
  return operation == ExprOp::MultiplyWrap || operation == ExprOp::Divide ||
         operation == ExprOp::AddSat || operation == ExprOp::AddSatUnsigned ||
         operation == ExprOp::SubSat || operation == ExprOp::BitAnd ||
         operation == ExprOp::BitOr || operation == ExprOp::BitXor ||
         operation == ExprOp::MulFixed || operation == ExprOp::MulFixedScaled ||
         operation == ExprOp::MulUnsignedFixed || operation == ExprOp::Atan2;
}

[[nodiscard]] constexpr bool
approximate_unary(const ExprOp operation) noexcept {
  return operation == ExprOp::Reciprocal || operation == ExprOp::Sqrt ||
         operation == ExprOp::Rsqrt || operation == ExprOp::Sin ||
         operation == ExprOp::Cos || operation == ExprOp::Tan ||
         operation == ExprOp::Exp || operation == ExprOp::Log;
}

[[nodiscard]] constexpr bool
approximate_binary(const ExprOp operation) noexcept {
  return operation == ExprOp::Divide || operation == ExprOp::Atan2;
}

[[nodiscard]] constexpr bool stored_format(const Type type,
                                           const FixedFormat format) noexcept {
  return static_cast<unsigned>(format.integer_bits) + format.fraction_bits ==
         bytes(type) * 8u;
}

[[nodiscard]] ExprRef append(const std::shared_ptr<ExprState> &state,
                             const ExprNode node) {
  if (state == nullptr) {
    return {};
  }
  if (!state->status) {
    return ExprRef{state, 0, node.type, node.fixed_format};
  }
  try {
    prepare_canonical(*state, state->nodes.size());
    const std::uint32_t existing = find_canonical(*state, node);
    if (existing != 0u) {
      return ExprRef{state, existing, node.type, node.fixed_format};
    }
    if (state->nodes.size() >= ExpressionCapacity) {
      set_error(state, Status::fail(Reason::ExpressionCapacity));
      return ExprRef{state, 0, node.type, node.fixed_format};
    }
    prepare_canonical(*state, state->nodes.size() + 1u);
    state->nodes.push_back(node);
    insert_canonical(*state, state->nodes.size() - 1u);
    state->canonical_nodes = state->nodes.size();
  } catch (const std::bad_alloc &) {
    set_error(state, Status::fail(Reason::ExpressionCapacity));
    return ExprRef{state, 0, node.type, node.fixed_format};
  }
  return ExprRef{state, static_cast<std::uint32_t>(state->nodes.size()),
                 node.type, node.fixed_format};
}

[[nodiscard]] bool valid(const ExprRef &value) noexcept {
  return value.state != nullptr && value.node > 0 &&
         value.node <= value.state->nodes.size();
}

} // namespace

std::shared_ptr<ExprState> make_expr() {
  try {
    return std::make_shared<ExprState>();
  } catch (const std::bad_alloc &) {
    return {};
  }
}

ExprRef input(const std::shared_ptr<ExprState> &state, const Type type,
              const std::uint32_t index, const FixedFormat fixed_format) {
  return append(state, ExprNode{
                           .operation = ExprOp::Input,
                           .type = type,
                           .fixed_format = fixed_format,
                           .left = index,
                       });
}

ExprRef constant(const std::shared_ptr<ExprState> &state, const Type type,
                 const std::uint64_t bits, const FixedFormat fixed_format) {
  return append(state, ExprNode{
                           .operation = ExprOp::Constant,
                           .type = type,
                           .fixed_format = fixed_format,
                           .bits = bits,
                       });
}

ExprRef index(const std::shared_ptr<ExprState> &state, const Type type) {
  return append(state, ExprNode{.operation = ExprOp::Index, .type = type});
}

ExprRef unary(const ExprOp operation, ExprRef value) {
  if (!valid(value)) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (fixed(value.type) && stored_unary(operation) &&
      !stored_format(value.type, value.fixed_format)) {
    set_error(value.state, Status::fail(Reason::FixedQuantizeRequired));
    return ExprRef{std::move(value.state), 0u, value.type, value.fixed_format};
  }
  FixedFormat format = value.fixed_format;
  if (fixed(value.type) && approximate_unary(operation)) {
    format.approximation = Approximation::Deterministic;
  }
  if (fixed(value.type) &&
      (operation == ExprOp::Negate || operation == ExprOp::Abs ||
       operation == ExprOp::AbsMagnitude)) {
    const unsigned width =
        static_cast<unsigned>(format.integer_bits) + format.fraction_bits;
    if (width >= 128u) {
      set_error(value.state, Status::fail(Reason::FixedPrecisionCapacity));
      return ExprRef{std::move(value.state), 0u, value.type,
                     value.fixed_format};
    }
    ++format.integer_bits;
  }
  return append(value.state, ExprNode{
                                 .operation = operation,
                                 .type = value.type,
                                 .fixed_format = format,
                                 .left = value.node,
                             });
}

ExprRef shift(const ExprOp operation, ExprRef value,
              const std::uint32_t amount) {
  if (!valid(value)) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (amount >= bytes(value.type) * 8u) {
    set_error(value.state, Status::fail(Reason::ShiftCountInvalid));
    return ExprRef{std::move(value.state), 0, value.type, value.fixed_format};
  }
  if (fixed(value.type) && !stored_format(value.type, value.fixed_format)) {
    set_error(value.state, Status::fail(Reason::FixedQuantizeRequired));
    return ExprRef{std::move(value.state), 0u, value.type, value.fixed_format};
  }
  return append(value.state, ExprNode{.operation = operation,
                                      .type = value.type,
                                      .fixed_format = value.fixed_format,
                                      .left = value.node,
                                      .immediate = amount});
}

ExprRef retype_expr(ExprRef value, const Type type) {
  if (!valid(value) || bytes(value.type) != bytes(type)) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, value.fixed_format};
  }
  value.type = type;
  return value;
}

ExprRef checked_ordinal_expr(ExprRef value, const Type type) {
  if (!valid(value) || fixed(value.type) || fixed(type) ||
      bytes(value.type) != bytes(type)) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, {}};
  }
  return append(value.state, ExprNode{
                                 .operation = ExprOp::CheckedOrdinal,
                                 .type = type,
                                 .left = value.node,
                             });
}

ExprRef boundary_mask_expr(ExprRef value, const Type type,
                           const FixedFormat fixed_format) {
  const bool source_integer = !fixed(value.type);
  const bool target_supported = type == Type::I32 || type == Type::U32 ||
                                type == Type::I64 || type == Type::U64 ||
                                type == Type::FixedLane32 ||
                                type == Type::FixedLane64;
  const bool format_valid = fixed(type) ? stored_format(type, fixed_format)
                                        : fixed_format == FixedFormat{};
  if (!valid(value) || !source_integer || !target_supported ||
      bytes(value.type) != bytes(type) || !format_valid) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::ExpressionTypeMismatch));
    }
    return ExprRef{std::move(value.state), 0u, type, fixed_format};
  }
  return append(value.state, ExprNode{
                                 .operation = ExprOp::BoundaryMask,
                                 .type = type,
                                 .fixed_format = fixed_format,
                                 .left = value.node,
                             });
}

ExprRef with_fixed_format(ExprRef value, const FixedFormat fixed_format) {
  if (!valid(value) || !fixed(value.type)) {
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
  if (!valid(value) && value.state != nullptr && !value.state->status) {
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  if (!valid(value) || !fixed(value.type) || !fixed(target) ||
      fixed_format.integer_bits == 0u || fixed_format.fraction_bits == 0u ||
      static_cast<unsigned>(fixed_format.integer_bits) +
              fixed_format.fraction_bits !=
          bytes(target) * 8u) {
    if (value.state != nullptr) {
      set_error(value.state, Status::fail(Reason::QuantizeFormatInvalid));
    }
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  if (value.fixed_format.approximation == Approximation::Deterministic &&
      fixed_format.approximation != Approximation::Deterministic) {
    set_error(value.state, Status::fail(Reason::FixedApproximationDowngrade));
    return ExprRef{std::move(value.state), 0u, target, fixed_format};
  }
  const ExprNode &source = value.state->nodes[value.node - 1u];
  if (source.operation == ExprOp::Quantize && value.type == target &&
      value.fixed_format == fixed_format) {
    return value;
  }
  return append(value.state, ExprNode{
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
  if (!valid(predicate)) {
    if (predicate.state != nullptr) {
      set_error(predicate.state, Status::fail(Reason::ExpressionInvalid));
    }
    return ExprRef{std::move(predicate.state), 0, output};
  }
  if (output != Type::U32 && output != Type::U64) {
    set_error(predicate.state, Status::fail(Reason::ExpressionTypeMismatch));
    return ExprRef{std::move(predicate.state), 0, output};
  }
  return append(predicate.state, ExprNode{
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
         (bytes(input) == sizeof(std::uint32_t) ||
          bytes(input) == sizeof(std::uint64_t)) &&
         expression.state->nodes[expression.node - 1u].operation ==
             ExprOp::Mask;
}

ExprRef binary(const ExprOp operation, ExprRef left, ExprRef right) {
  if (!valid(left) || !valid(right) || left.state != right.state ||
      left.type != right.type) {
    if (left.state != nullptr) {
      set_error(left.state, Status::fail(Reason::ExpressionContextMismatch));
    }
    return ExprRef{std::move(left.state), 0, left.type, left.fixed_format};
  }
  FixedFormat format = left.fixed_format;
  if (fixed(left.type)) {
    if (left.fixed_format.integer_bits == 0u ||
        right.fixed_format.integer_bits == 0u ||
        left.fixed_format.rounding != right.fixed_format.rounding ||
        left.fixed_format.overflow != right.fixed_format.overflow) {
      set_error(left.state, Status::fail(Reason::FixedFormatMismatch));
      return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
    }
    if (stored_binary(operation) &&
        (!stored_format(left.type, left.fixed_format) ||
         !stored_format(right.type, right.fixed_format) ||
         left.fixed_format.integer_bits != right.fixed_format.integer_bits ||
         left.fixed_format.fraction_bits != right.fixed_format.fraction_bits)) {
      set_error(left.state, Status::fail(Reason::FixedQuantizeRequired));
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
        set_error(left.state, Status::fail(Reason::FixedPrecisionCapacity));
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
        set_error(left.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    } else {
      const unsigned integer = std::max(left_integer, right_integer);
      const unsigned fraction = std::max(left_fraction, right_fraction);
      if (integer + fraction > 128u) {
        set_error(left.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(left.state), 0u, left.type, left.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    }
    if (approximate_binary(operation)) {
      format.approximation = Approximation::Deterministic;
    }
  }
  return append(left.state, ExprNode{
                                .operation = operation,
                                .type = left.type,
                                .fixed_format = format,
                                .left = left.node,
                                .right = right.node,
                            });
}

ExprRef ternary(const ExprOp operation, ExprRef first, ExprRef second,
                ExprRef third) {
  if (!valid(first) || !valid(second) || !valid(third) ||
      first.state != second.state || first.state != third.state ||
      second.type != third.type) {
    if (first.state != nullptr) {
      set_error(first.state, Status::fail(Reason::ExpressionContextMismatch));
    }
    return ExprRef{std::move(first.state), 0, second.type, second.fixed_format};
  }
  FixedFormat format = second.fixed_format;
  if (fixed(second.type)) {
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
      set_error(first.state, Status::fail(Reason::FixedFormatMismatch));
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
        set_error(first.state, Status::fail(Reason::FixedPrecisionCapacity));
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
        set_error(first.state, Status::fail(Reason::FixedPrecisionCapacity));
        return ExprRef{std::move(first.state), 0u, second.type,
                       second.fixed_format};
      }
      format.integer_bits = static_cast<unsigned char>(integer);
      format.fraction_bits = static_cast<unsigned char>(fraction);
    }
  }
  return append(first.state, ExprNode{
                                 .operation = operation,
                                 .type = second.type,
                                 .fixed_format = format,
                                 .left = first.node,
                                 .right = second.node,
                                 .third = third.node,
                             });
}

} // namespace rund::compute::detail
