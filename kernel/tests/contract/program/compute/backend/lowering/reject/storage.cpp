#include "contract/program/compute/backend/lowering/reject/model.hpp"
#include "test/assert.hpp"

#include <array>
#include <string_view>

namespace program_compute_contract::lowering_reject {
namespace {

using namespace backend_lowering_support;

[[nodiscard]] auto Widened32() {
  i32 input[1]{};
  i32 output[1]{};
  const auto body =
      rund::compute_dsl::bind(1u)
          .fixed<16u, 16u, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .read<"input">(input)
          .write<"output">(output);
  return rund::compute_dsl::def("forged-storage-gate")
      .on(body)
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        const auto output = b.template write<"output">();
        output[i] = input[i] + input[i];
      });
}

[[nodiscard]] auto Widened64() {
  rund::kernel::i64 input[1]{};
  rund::kernel::i64 output[1]{};
  const auto body =
      rund::compute_dsl::bind(1u)
          .fixed<32u, 32u, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .read<"input">(input)
          .write<"output">(output);
  return rund::compute_dsl::def("forged-storage-gate-64")
      .on(body)
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        const auto output = b.template write<"output">();
        output[i] = input[i] + input[i];
      });
}

template <class Operation>
[[nodiscard]] bool OrderRejects(const Operation &op) {
  if (!op.ok()) {
    return false;
  }
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
  if (!parsed.ok) {
    return false;
  }
  const auto target = FindNode(parsed, rund::kernel::IrOp::Add);
  const auto read = FindNode(parsed, rund::kernel::IrOp::Read);
  if (target == 0u || read == 0u) {
    return false;
  }
  constexpr std::array ops{
      OpArity{rund::kernel::IrOp::MinUnsigned, 2u},
      OpArity{rund::kernel::IrOp::MaxUnsigned, 2u},
      OpArity{rund::kernel::IrOp::ClampUnsigned, 3u},
      OpArity{rund::kernel::IrOp::LtUnsigned, 2u},
      OpArity{rund::kernel::IrOp::LeUnsigned, 2u},
      OpArity{rund::kernel::IrOp::GtUnsigned, 2u},
      OpArity{rund::kernel::IrOp::GeUnsigned, 2u},
  };
  for (const auto spec : ops) {
    auto forged = op.ir();
    if (!SetNode(forged.canonical_bytes, target, spec.op, read,
                 spec.arity >= 2u ? read : 0u, spec.arity >= 3u ? read : 0u)) {
      return false;
    }
    forged = RehashIr(std::move(forged));
    const auto forged_parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(forged);
    if (!forged_parsed.ok ||
        static_cast<rund::kernel::IrOp>(forged_parsed.nodes[target - 1u].op) !=
            spec.op ||
        rund::kernel::compute_lowering_detail::DomainForMode(
            op.ir().scalar, forged_parsed.scalar_mode) !=
            rund::kernel::ComputeDomain::Fixed ||
        rund::kernel::IrOpDomainValid(spec.op,
                                      rund::kernel::ComputeDomain::Fixed) ||
        !rejection_support::Rejects(forged, "compute_ir_node_invalid")) {
      return false;
    }
  }
  return true;
}

int Gate() {
  const auto op = Widened32();
  TEST_ASSERT(op.ok());
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
  TEST_ASSERT(parsed.ok);
  const auto read = FindNode(parsed, rund::kernel::IrOp::Read);
  const auto widened = FindNode(parsed, rund::kernel::IrOp::Add);
  const auto quantize = FindNode(parsed, rund::kernel::IrOp::Quantize);
  TEST_ASSERT(read != 0u && widened != 0u && quantize != 0u);

  const auto rejects =
      [&](const rund::kernel::IrOp kind, const rund::kernel::u32 rhs,
          const rund::kernel::u32 aux, const std::string_view expected) {
        auto forged = parsed;
        auto &node = forged.nodes[quantize - 1u];
        node.op = static_cast<rund::kernel::u8>(kind);
        node.lhs = widened;
        node.rhs = rhs;
        node.aux = aux;
        const char *const reason =
            rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                forged, op.ir().scalar);
        return reason != nullptr && std::string_view{reason} == expected;
      };

  constexpr std::array unary{
      rund::kernel::IrOp::BitNot, rund::kernel::IrOp::NegPositiveFixed,
      rund::kernel::IrOp::Recip,  rund::kernel::IrOp::Sqrt,
      rund::kernel::IrOp::Rsqrt,  rund::kernel::IrOp::Sin,
      rund::kernel::IrOp::Cos,    rund::kernel::IrOp::Tan,
      rund::kernel::IrOp::Exp,    rund::kernel::IrOp::Log,
  };
  for (const auto kind : unary) {
    TEST_ASSERT(rejects(kind, 0u, 0u, "compute_ir_quantize_required"));
  }

  constexpr std::array binary{
      rund::kernel::IrOp::MulWrap,        rund::kernel::IrOp::BitAnd,
      rund::kernel::IrOp::BitOr,          rund::kernel::IrOp::BitXor,
      rund::kernel::IrOp::AddSat,         rund::kernel::IrOp::AddSatUnsigned,
      rund::kernel::IrOp::SubSat,         rund::kernel::IrOp::MulFixed,
      rund::kernel::IrOp::MulFixedScaled, rund::kernel::IrOp::MulUnsignedFixed,
      rund::kernel::IrOp::DivFixed,       rund::kernel::IrOp::Atan2,
  };
  for (const auto kind : binary) {
    TEST_ASSERT(rejects(kind, read, 0u, "compute_ir_quantize_required"));
  }
  for (const auto kind :
       {rund::kernel::IrOp::ShlConst, rund::kernel::IrOp::ShrLogicalConst,
        rund::kernel::IrOp::ShrArithmeticConst}) {
    TEST_ASSERT(rejects(kind, 0u, 1u, "compute_ir_quantize_required"));
  }
  for (const auto kind :
       {rund::kernel::IrOp::MinUnsigned, rund::kernel::IrOp::MaxUnsigned,
        rund::kernel::IrOp::LtUnsigned, rund::kernel::IrOp::LeUnsigned,
        rund::kernel::IrOp::GtUnsigned, rund::kernel::IrOp::GeUnsigned}) {
    TEST_ASSERT(rejects(kind, read, 0u, "compute_ir_node_invalid"));
  }
  TEST_ASSERT(rejects(rund::kernel::IrOp::ClampUnsigned, read, read,
                      "compute_ir_node_invalid"));
  for (const auto kind :
       {rund::kernel::IrOp::DivSigned, rund::kernel::IrOp::DivUnsigned}) {
    TEST_ASSERT(rejects(kind, read, 0u, "compute_ir_node_invalid"));
  }
  TEST_ASSERT(OrderRejects(Widened32()));
  TEST_ASSERT(OrderRejects(Widened64()));
  return 0;
}

} // namespace

int Storage() { return Gate(); }

} // namespace program_compute_contract::lowering_reject
