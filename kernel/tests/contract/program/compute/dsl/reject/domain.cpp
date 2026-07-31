#include "contract/program/compute/dsl/reject/model.hpp"
#include "test/assert.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace program_compute_contract::dsl_reject {
namespace {

using rejection_support::Mode;

inline constexpr std::array kFixedOnlyOps{
    rund::kernel::IrOp::Quantize,
    rund::kernel::IrOp::NegPositiveFixed,
    rund::kernel::IrOp::MulFixed,
    rund::kernel::IrOp::MulFixedScaled,
    rund::kernel::IrOp::MulUnsignedFixed,
    rund::kernel::IrOp::MulAddFixed,
    rund::kernel::IrOp::DivFixed,
    rund::kernel::IrOp::Recip,
    rund::kernel::IrOp::Sqrt,
    rund::kernel::IrOp::Rsqrt,
    rund::kernel::IrOp::Sin,
    rund::kernel::IrOp::Cos,
    rund::kernel::IrOp::Tan,
    rund::kernel::IrOp::Exp,
    rund::kernel::IrOp::Log,
    rund::kernel::IrOp::Atan2,
};

[[nodiscard]] consteval bool FixedSetExact() {
  for (rund::kernel::u32 raw = 0u;
       raw <= static_cast<rund::kernel::u32>(rund::kernel::IrOp::Quantize);
       ++raw) {
    const auto op = static_cast<rund::kernel::IrOp>(raw);
    const bool expected = std::find(kFixedOnlyOps.begin(), kFixedOnlyOps.end(),
                                    op) != kFixedOnlyOps.end();
    if (rund::kernel::FixedOnlyOp(op) != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] consteval bool MergeExact() {
  using rund::kernel::ComputeDomain;
  constexpr auto invalid = static_cast<ComputeDomain>(0u);
  constexpr std::array domains{ComputeDomain::I32, ComputeDomain::U32,
                               ComputeDomain::I64, ComputeDomain::U64,
                               ComputeDomain::Fixed};
  constexpr std::array<std::array<ComputeDomain, 5u>, 5u> expected{{
      {ComputeDomain::I32, ComputeDomain::U32, ComputeDomain::I64,
       ComputeDomain::U64, invalid},
      {ComputeDomain::U32, ComputeDomain::U32, ComputeDomain::I64,
       ComputeDomain::U64, invalid},
      {ComputeDomain::I64, ComputeDomain::I64, ComputeDomain::I64,
       ComputeDomain::U64, invalid},
      {ComputeDomain::U64, ComputeDomain::U64, ComputeDomain::U64,
       ComputeDomain::U64, invalid},
      {invalid, invalid, invalid, invalid, ComputeDomain::Fixed},
  }};
  for (std::size_t lhs = 0u; lhs < domains.size(); ++lhs) {
    for (std::size_t rhs = 0u; rhs < domains.size(); ++rhs) {
      if (rund::kernel::MergeComputeDomains(domains[lhs], domains[rhs]) !=
          expected[lhs][rhs]) {
        return false;
      }
    }
  }
  return true;
}

static_assert(FixedSetExact());
static_assert(MergeExact());
static_assert(rund::kernel::IrOpDomainValid(rund::kernel::IrOp::AddSat,
                                            rund::kernel::ComputeDomain::I32));
static_assert(rund::kernel::IrOpDomainValid(rund::kernel::IrOp::AddSatUnsigned,
                                            rund::kernel::ComputeDomain::U32));
static_assert(rund::kernel::IrOpDomainValid(rund::kernel::IrOp::AbsMagnitude,
                                            rund::kernel::ComputeDomain::I64));

struct OrderPair final {
  rund::kernel::IrOp base;
  rund::kernel::IrOp canonical;
};

template <class T> [[nodiscard]] auto OrderOp(const rund::kernel::IrOp op) {
  T input[1]{};
  T output[1]{};
  const auto body = [&] {
    if constexpr (sizeof(T) == sizeof(rund::kernel::u32)) {
      return rund::compute_dsl::bind(1u)
          .u32()
          .read<"input">(input)
          .template write<"output">(output);
    } else {
      return rund::compute_dsl::bind(1u)
          .u64()
          .read<"input">(input)
          .template write<"output">(output);
    }
  }();
  return rund::compute_dsl::def("canonical-unsigned-order")
      .on(body)
      .map([op](auto index, auto bindings) {
        const auto input = bindings.template read<"input">();
        const auto output = bindings.template write<"output">();
        const auto value = input[index];
        const auto result =
            op == rund::kernel::IrOp::Clamp ||
                    op == rund::kernel::IrOp::ClampUnsigned
                ? rund::compute_dsl::detail::Ternary(op, value, value, value)
                : rund::compute_dsl::detail::Binary(op, value, value);
        output[index] = result;
      });
}

template <class T> [[nodiscard]] bool OrderStable() {
  constexpr std::array pairs{
      OrderPair{rund::kernel::IrOp::Min, rund::kernel::IrOp::MinUnsigned},
      OrderPair{rund::kernel::IrOp::Max, rund::kernel::IrOp::MaxUnsigned},
      OrderPair{rund::kernel::IrOp::Clamp, rund::kernel::IrOp::ClampUnsigned},
      OrderPair{rund::kernel::IrOp::Lt, rund::kernel::IrOp::LtUnsigned},
      OrderPair{rund::kernel::IrOp::Le, rund::kernel::IrOp::LeUnsigned},
      OrderPair{rund::kernel::IrOp::Gt, rund::kernel::IrOp::GtUnsigned},
      OrderPair{rund::kernel::IrOp::Ge, rund::kernel::IrOp::GeUnsigned},
  };
  for (const auto pair : pairs) {
    const auto base = OrderOp<T>(pair.base);
    const auto canonical = OrderOp<T>(pair.canonical);
    if (!base.ok() || !canonical.ok() ||
        base.ir().canonical_bytes != canonical.ir().canonical_bytes ||
        base.ir().op_hash_hi != canonical.ir().op_hash_hi ||
        base.ir().op_hash_lo != canonical.ir().op_hash_lo) {
      return false;
    }
  }
  return true;
}

int FixedOnly() {
  constexpr std::array modes{Mode::I32, Mode::U32, Mode::I64, Mode::U64};
  constexpr std::array unary_ops{
      rund::kernel::IrOp::Quantize, rund::kernel::IrOp::NegPositiveFixed,
      rund::kernel::IrOp::Recip,    rund::kernel::IrOp::Sqrt,
      rund::kernel::IrOp::Rsqrt,    rund::kernel::IrOp::Sin,
      rund::kernel::IrOp::Cos,      rund::kernel::IrOp::Tan,
      rund::kernel::IrOp::Exp,      rund::kernel::IrOp::Log,
  };
  constexpr std::array binary_ops{
      rund::kernel::IrOp::MulFixed,
      rund::kernel::IrOp::MulFixedScaled,
      rund::kernel::IrOp::MulUnsignedFixed,
      rund::kernel::IrOp::DivFixed,
      rund::kernel::IrOp::Atan2,
  };
  for (const auto mode : modes) {
    for (const auto op : unary_ops) {
      TEST_ASSERT(Rejects(IntegerOp(op, mode, Arity::Unary)));
    }
    for (const auto op : binary_ops) {
      TEST_ASSERT(Rejects(IntegerOp(op, mode, Arity::Binary)));
    }
    TEST_ASSERT(Rejects(
        IntegerOp(rund::kernel::IrOp::MulAddFixed, mode, Arity::Ternary)));
  }
  return 0;
}

int Divide() {
  for (const auto mode : {Mode::I32, Mode::I64}) {
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::DivSigned, mode, Arity::Binary).ok);
    TEST_ASSERT(Rejects(
        IntegerOp(rund::kernel::IrOp::DivUnsigned, mode, Arity::Binary)));
  }
  for (const auto mode : {Mode::U32, Mode::U64}) {
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::DivUnsigned, mode, Arity::Binary).ok);
    TEST_ASSERT(
        Rejects(IntegerOp(rund::kernel::IrOp::DivSigned, mode, Arity::Binary)));
  }
  return 0;
}

int Signedness() {
  constexpr std::array signed_modes{Mode::I32, Mode::I64};
  constexpr std::array unsigned_modes{Mode::U32, Mode::U64};
  constexpr std::array signed_unary{
      rund::kernel::IrOp::Abs,
      rund::kernel::IrOp::AbsMagnitude,
      rund::kernel::IrOp::Sign,
  };
  constexpr std::array signed_binary{
      rund::kernel::IrOp::AddSat,
      rund::kernel::IrOp::SubSat,
  };
  constexpr std::array unsigned_binary{
      rund::kernel::IrOp::MinUnsigned, rund::kernel::IrOp::MaxUnsigned,
      rund::kernel::IrOp::LtUnsigned,  rund::kernel::IrOp::LeUnsigned,
      rund::kernel::IrOp::GtUnsigned,  rund::kernel::IrOp::GeUnsigned,
  };
  constexpr std::array generic_binary{
      rund::kernel::IrOp::Min, rund::kernel::IrOp::Max, rund::kernel::IrOp::Lt,
      rund::kernel::IrOp::Le,  rund::kernel::IrOp::Gt,  rund::kernel::IrOp::Ge,
  };

  for (const auto mode : signed_modes) {
    for (const auto op : signed_unary) {
      TEST_ASSERT(IntegerOp(op, mode, Arity::Unary).ok);
    }
    for (const auto op : signed_binary) {
      TEST_ASSERT(IntegerOp(op, mode, Arity::Binary).ok);
    }
    TEST_ASSERT(Rejects(
        IntegerOp(rund::kernel::IrOp::AddSatUnsigned, mode, Arity::Binary)));
    for (const auto op : unsigned_binary) {
      TEST_ASSERT(Rejects(IntegerOp(op, mode, Arity::Binary)));
    }
    TEST_ASSERT(Rejects(
        IntegerOp(rund::kernel::IrOp::ClampUnsigned, mode, Arity::Ternary)));
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::ShrArithmeticConst, mode, Arity::Shift)
            .ok);
  }

  for (const auto mode : unsigned_modes) {
    for (const auto op : signed_unary) {
      TEST_ASSERT(Rejects(IntegerOp(op, mode, Arity::Unary)));
    }
    for (const auto op : signed_binary) {
      TEST_ASSERT(Rejects(IntegerOp(op, mode, Arity::Binary)));
    }
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::AddSatUnsigned, mode, Arity::Binary).ok);
    for (const auto op : unsigned_binary) {
      TEST_ASSERT(IntegerOp(op, mode, Arity::Binary).ok);
    }
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::ClampUnsigned, mode, Arity::Ternary).ok);
    TEST_ASSERT(Rejects(
        IntegerOp(rund::kernel::IrOp::ShrArithmeticConst, mode, Arity::Shift)));
  }

  constexpr std::array all_modes{Mode::I32, Mode::U32, Mode::I64, Mode::U64};
  for (const auto mode : all_modes) {
    TEST_ASSERT(IntegerOp(rund::kernel::IrOp::Neg, mode, Arity::Unary).ok);
    for (const auto op : generic_binary) {
      TEST_ASSERT(IntegerOp(op, mode, Arity::Binary).ok);
    }
    TEST_ASSERT(IntegerOp(rund::kernel::IrOp::Clamp, mode, Arity::Ternary).ok);
    TEST_ASSERT(IntegerOp(rund::kernel::IrOp::ShlConst, mode, Arity::Shift).ok);
    TEST_ASSERT(
        IntegerOp(rund::kernel::IrOp::ShrLogicalConst, mode, Arity::Shift).ok);
  }

  constexpr std::array canonical{
      std::pair{rund::kernel::IrOp::Min, rund::kernel::IrOp::MinUnsigned},
      std::pair{rund::kernel::IrOp::Max, rund::kernel::IrOp::MaxUnsigned},
      std::pair{rund::kernel::IrOp::Lt, rund::kernel::IrOp::LtUnsigned},
      std::pair{rund::kernel::IrOp::Le, rund::kernel::IrOp::LeUnsigned},
      std::pair{rund::kernel::IrOp::Gt, rund::kernel::IrOp::GtUnsigned},
      std::pair{rund::kernel::IrOp::Ge, rund::kernel::IrOp::GeUnsigned},
  };
  for (const auto mode : unsigned_modes) {
    for (const auto& [requested, expected] : canonical) {
      const auto outcome = IntegerOp(requested, mode, Arity::Binary);
      TEST_ASSERT(outcome.ok && outcome.last == expected);
    }
    const auto clamp =
        IntegerOp(rund::kernel::IrOp::Clamp, mode, Arity::Ternary);
    TEST_ASSERT(clamp.ok && clamp.last == rund::kernel::IrOp::ClampUnsigned);
  }

  for (const auto op : signed_unary) {
    TEST_ASSERT(FixedOp(op, Arity::Unary).ok);
  }
  for (const auto op : signed_binary) {
    TEST_ASSERT(FixedOp(op, Arity::Binary).ok);
  }
  TEST_ASSERT(FixedOp(rund::kernel::IrOp::AddSatUnsigned, Arity::Binary).ok);
  for (const auto op : unsigned_binary) {
    TEST_ASSERT(Rejects(FixedOp(op, Arity::Binary)));
  }
  for (const auto op : generic_binary) {
    TEST_ASSERT(FixedOp(op, Arity::Binary).ok);
  }
  TEST_ASSERT(FixedOp(rund::kernel::IrOp::Clamp, Arity::Ternary).ok);
  TEST_ASSERT(
      Rejects(FixedOp(rund::kernel::IrOp::ClampUnsigned, Arity::Ternary)));
  TEST_ASSERT(FixedOp(rund::kernel::IrOp::ShrArithmeticConst, Arity::Shift).ok);
  TEST_ASSERT(OrderStable<rund::kernel::u32>());
  TEST_ASSERT(OrderStable<rund::kernel::u64>());
  return 0;
}

} // namespace

int Domain() {
  if (FixedOnly() != 0 || Divide() != 0) {
    return 1;
  }
  return Signedness();
}

} // namespace program_compute_contract::dsl_reject
