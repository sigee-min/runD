#include "internal.hpp"

#include "../../../hash/fnv.hpp"
#include "../../type.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail::expression_build {
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

} // namespace

void set_error(const std::shared_ptr<ExprState> &state, Status status) {
  if (state != nullptr && state->status) {
    state->status = std::move(status);
  }
}

bool stored_unary(const ExprOp operation) noexcept {
  return operation == ExprOp::NegPositiveFixed || operation == ExprOp::BitNot ||
         operation == ExprOp::Reciprocal || operation == ExprOp::Sqrt ||
         operation == ExprOp::Rsqrt || operation == ExprOp::Sin ||
         operation == ExprOp::Cos || operation == ExprOp::Tan ||
         operation == ExprOp::Exp || operation == ExprOp::Log;
}

bool stored_binary(const ExprOp operation) noexcept {
  return operation == ExprOp::MultiplyWrap || operation == ExprOp::Divide ||
         operation == ExprOp::AddSat || operation == ExprOp::AddSatUnsigned ||
         operation == ExprOp::SubSat || operation == ExprOp::BitAnd ||
         operation == ExprOp::BitOr || operation == ExprOp::BitXor ||
         operation == ExprOp::MulFixed || operation == ExprOp::MulFixedScaled ||
         operation == ExprOp::MulUnsignedFixed || operation == ExprOp::Atan2;
}

bool approximate_unary(const ExprOp operation) noexcept {
  return operation == ExprOp::Reciprocal || operation == ExprOp::Sqrt ||
         operation == ExprOp::Rsqrt || operation == ExprOp::Sin ||
         operation == ExprOp::Cos || operation == ExprOp::Tan ||
         operation == ExprOp::Exp || operation == ExprOp::Log;
}

bool approximate_binary(const ExprOp operation) noexcept {
  return operation == ExprOp::Divide || operation == ExprOp::Atan2;
}

bool stored_format(const Type type, const FixedFormat format) noexcept {
  return static_cast<unsigned>(format.integer_bits) + format.fraction_bits ==
         type_bytes(type) * 8u;
}

ExprRef append(const std::shared_ptr<ExprState> &state, const ExprNode node) {
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

bool valid(const ExprRef &value) noexcept {
  return value.state != nullptr && value.node > 0 &&
         value.node <= value.state->nodes.size();
}

} // namespace rund::compute::detail::expression_build
