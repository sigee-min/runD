#pragma once

#include <rund/compute/abi/expression.hpp>

#include <kernel/program/compute/limit.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::compute::detail {

inline constexpr std::size_t ExpressionCapacity =
    static_cast<std::size_t>(kernel::kMaxComputeNodeCount);
static_assert(ExpressionCapacity <= std::numeric_limits<std::uint16_t>::max());

inline constexpr std::uint8_t InvalidArity = 0xffu;

[[nodiscard]] std::uint8_t expr_arity(ExprOp operation) noexcept;
[[nodiscard]] bool supported(ExprOp operation) noexcept;

struct ExprNode final {
  ExprOp operation{ExprOp::Input};
  Type type{Type::I32};
  FixedFormat fixed_format{};
  std::uint32_t left{};
  std::uint32_t right{};
  std::uint32_t third{};
  std::uint32_t immediate{};
  std::uint64_t bits{};

  [[nodiscard]] constexpr bool
  operator==(const ExprNode &) const noexcept = default;
};

struct ExprState final {
  std::vector<ExprNode> nodes;
  // Open-addressed node-index table used only while constructing an
  // expression. Slots store node index + 1 so the node vector remains the
  // single semantic owner and insertion order remains canonical.
  std::vector<std::uint16_t> canonical_slots;
  std::size_t canonical_nodes{};
  Status status{Status::success()};
};

} // namespace rund::compute::detail
