#pragma once

#include "../step.hpp"
#include "../../expression/state.hpp"
#include "../../fixed/format.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/ir.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rund::compute::detail {

using KernelExpr = compute_dsl::detail::Expr;

struct InputFixedFormats final {
  std::array<FixedFormat, MaxMapInputs> values{};
  std::uint16_t present = 0u;

  [[nodiscard]] FixedFormat get(const std::size_t input,
                                const FixedFormat fallback) const noexcept {
    if (input >= values.size()) {
      return fallback;
    }
    const auto bit = static_cast<std::uint16_t>(std::uint16_t{1u} << input);
    return (present & bit) == 0u ? fallback : values[input];
  }
};

[[nodiscard]] constexpr compute_dsl::detail::ScalarMode
numeric_mode(const Type type) noexcept {
  switch (type) {
  case Type::I32:
    return compute_dsl::detail::ScalarMode::I32;
  case Type::U32:
    return compute_dsl::detail::ScalarMode::U32;
  case Type::I64:
    return compute_dsl::detail::ScalarMode::I64;
  case Type::U64:
    return compute_dsl::detail::ScalarMode::U64;
  case Type::FixedLane32:
    return compute_dsl::detail::ScalarMode::FixedLane32;
  case Type::FixedLane64:
    return compute_dsl::detail::ScalarMode::FixedLane64;
  }
  return compute_dsl::detail::ScalarMode::Unspecified;
}

[[nodiscard]] std::optional<kernel::IrOp> binary_op(ExprOp operation,
                                                    Type type) noexcept;
[[nodiscard]] std::optional<kernel::IrOp> unary_op(ExprOp operation) noexcept;

[[nodiscard]] std::optional<KernelExpr>
replay(const ExprRef &expression, std::span<const KernelExpr> inputs,
       KernelExpr constant_anchor, KernelExpr logical_index,
       std::vector<KernelExpr> &values, std::vector<std::uint8_t> &state);

[[nodiscard]] bool expressions_ok(std::span<const Type> outputs,
                                  std::span<const Type> inputs,
                                  std::span<const ExprRef> expressions,
                                  InputFixedFormats &input_formats) noexcept;
[[nodiscard]] bool
fixed_output_missing_quantize(std::span<const Type> outputs,
                              std::span<const ExprRef> expressions) noexcept;

[[nodiscard]] compute_dsl::ComputeOp
build_dynamic_op(std::size_t count, std::span<const Type> outputs,
                 std::span<const Type> inputs,
                 std::span<const ExprRef> expressions,
                 const InputFixedFormats &input_formats,
                 std::span<const MapRead> reads);

} // namespace rund::compute::detail
