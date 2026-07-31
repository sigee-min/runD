#pragma once

#include "contract/program/compute/dsl/reject/local.hpp"

namespace program_compute_contract::dsl_reject {

enum class Arity {
  Unary,
  Binary,
  Ternary,
  Shift,
};

struct Outcome final {
  bool ok = false;
  const char *reason = "";
  rund::kernel::IrOp last = {};
};

[[nodiscard]] inline Outcome IntegerOp(const rund::kernel::IrOp op,
                                       const rejection_support::Mode mode,
                                       const Arity arity) {
  using namespace rund::compute_dsl::detail;
  const auto bindings = rejection_support::IntegerBindings(mode);
  BuildContext context{bindings, mode};
  const auto value = DynamicRead(context, 0u);
  switch (arity) {
  case Arity::Unary:
    (void)Unary(op, value);
    break;
  case Arity::Binary:
    (void)Binary(op, value, value);
    break;
  case Arity::Ternary:
    (void)Ternary(op, value, value, value);
    break;
  case Arity::Shift:
    (void)ConstShift(op, value, 1u);
    break;
  }
  return Outcome{
      .ok = context.ok(),
      .reason = context.reason(),
      .last = context.nodes().empty() ? rund::kernel::IrOp{}
                                      : context.nodes().back().op,
  };
}

[[nodiscard]] inline Outcome FixedOp(const rund::kernel::IrOp op,
                                     const Arity arity,
                                     const bool formatted = false,
                                     const bool widened = false) {
  using namespace rund::compute_dsl::detail;
  rund::kernel::i32 input[1]{};
  const auto body =
      rund::compute_dsl::bind(1u).fixed<16, 16>().read<"input">(input);
  BuildContext context{body.bindings(), rejection_support::Mode::FixedLane32,
                       body.fixed_format()};
  const auto stored = DynamicRead(context, 0u);
  const auto value =
      widened ? Binary(rund::kernel::IrOp::Add, stored, stored) : stored;
  switch (arity) {
  case Arity::Unary:
    if (formatted) {
      (void)UnaryFormatted(op, value, body.fixed_format());
    } else {
      (void)Unary(op, value);
    }
    break;
  case Arity::Binary:
    if (formatted) {
      (void)BinaryFormatted(op, value, stored, body.fixed_format());
    } else {
      (void)Binary(op, value, stored);
    }
    break;
  case Arity::Ternary:
    (void)Ternary(op, value, stored, stored);
    break;
  case Arity::Shift:
    (void)ConstShift(op, value, 1u);
    break;
  }
  return Outcome{
      .ok = context.ok(),
      .reason = context.reason(),
      .last = context.nodes().empty() ? rund::kernel::IrOp{}
                                      : context.nodes().back().op,
  };
}

[[nodiscard]] inline bool
Rejects(const Outcome outcome,
        const std::string_view reason = "compute_value_invalid") noexcept {
  return !outcome.ok && std::string_view{outcome.reason} == reason;
}

} // namespace program_compute_contract::dsl_reject
