#include "contract/program/compute/backend/lowering/reject/model.hpp"
#include "test/assert.hpp"

#include <array>

namespace program_compute_contract::lowering_reject {
namespace {

int LargeChain() {
  using namespace rund::kernel::compute_lowering_detail;
  constexpr std::size_t kOperationCount =
      rund::kernel::kMaxComputeNodeCount - 3u;
  ParsedIR parsed{
      .name = "",
      .scalar_mode = DomainModeFor(rund::kernel::ComputeScalar::Lane32,
                                   rund::kernel::ComputeDomain::I32),
      .fixed_format = {},
      .bindings =
          {
              ParsedBinding{
                  .kind = kReadBindingKind,
                  .numeric_mode =
                      DomainModeFor(rund::kernel::ComputeScalar::Lane32,
                                    rund::kernel::ComputeDomain::I32),
                  .name = "input",
                  .element_bytes = sizeof(rund::kernel::i32),
                  .floating_point_param = false,
                  .value_bytes = {},
              },
              ParsedBinding{
                  .kind = kWriteBindingKind,
                  .numeric_mode =
                      DomainModeFor(rund::kernel::ComputeScalar::Lane32,
                                    rund::kernel::ComputeDomain::I32),
                  .name = "output",
                  .element_bytes = sizeof(rund::kernel::i32),
                  .floating_point_param = false,
                  .value_bytes = {},
              },
          },
      .nodes = {},
      .ok = true,
      .reason = "ok",
  };
  parsed.nodes.reserve(kOperationCount + 3u);
  parsed.nodes.push_back(
      ParsedNode{.op = static_cast<rund::kernel::u8>(rund::kernel::IrOp::Read),
                 .aux = 0u});
  parsed.nodes.push_back(ParsedNode{
      .op = static_cast<rund::kernel::u8>(rund::kernel::IrOp::Constant),
      .lhs = 1u,
  });
  rund::kernel::u32 value = 1u;
  for (std::size_t index = 0u; index < kOperationCount; ++index) {
    parsed.nodes.push_back(ParsedNode{
        .op = static_cast<rund::kernel::u8>(rund::kernel::IrOp::Add),
        .lhs = value,
        .rhs = 2u,
    });
    value = static_cast<rund::kernel::u32>(parsed.nodes.size());
  }
  parsed.nodes.push_back(ParsedNode{
      .op = static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write),
      .lhs = value,
      .aux = 1u,
  });
  TEST_ASSERT(ValidateLowerableIR(
                  parsed, rund::kernel::ComputeScalar::Lane32) == nullptr);
  return 0;
}

int FixedOnly() {
  constexpr std::array ops{
      OpArity{rund::kernel::IrOp::Quantize, 1u},
      OpArity{rund::kernel::IrOp::NegPositiveFixed, 1u},
      OpArity{rund::kernel::IrOp::MulFixed, 2u},
      OpArity{rund::kernel::IrOp::MulFixedScaled, 2u},
      OpArity{rund::kernel::IrOp::MulUnsignedFixed, 2u},
      OpArity{rund::kernel::IrOp::MulAddFixed, 3u},
      OpArity{rund::kernel::IrOp::DivFixed, 2u},
      OpArity{rund::kernel::IrOp::Recip, 1u},
      OpArity{rund::kernel::IrOp::Sqrt, 1u},
      OpArity{rund::kernel::IrOp::Rsqrt, 1u},
      OpArity{rund::kernel::IrOp::Sin, 1u},
      OpArity{rund::kernel::IrOp::Cos, 1u},
      OpArity{rund::kernel::IrOp::Tan, 1u},
      OpArity{rund::kernel::IrOp::Exp, 1u},
      OpArity{rund::kernel::IrOp::Log, 1u},
      OpArity{rund::kernel::IrOp::Atan2, 2u},
  };
  static_assert(ops.size() == 16u);
  for (const auto mode : kIntegerDomains) {
    for (const auto spec : ops) {
      TEST_ASSERT(rejection_support::Rejects(
          IntegerIr(spec.op, spec.arity, mode), "compute_ir_node_invalid"));
    }
  }
  return 0;
}

int Divide() {
  for (const auto mode : kIntegerDomains) {
    const auto valid = Signed(mode.domain) ? rund::kernel::IrOp::DivSigned
                                           : rund::kernel::IrOp::DivUnsigned;
    const auto invalid = Signed(mode.domain) ? rund::kernel::IrOp::DivUnsigned
                                             : rund::kernel::IrOp::DivSigned;
    TEST_ASSERT(rejection_support::Accepts(IntegerIr(valid, 2u, mode)));
    TEST_ASSERT(rejection_support::Rejects(IntegerIr(invalid, 2u, mode),
                                           "compute_ir_node_invalid"));
  }
  return 0;
}

int Signedness() {
  constexpr std::array signed_ops{
      OpArity{rund::kernel::IrOp::Abs, 1u},
      OpArity{rund::kernel::IrOp::AbsMagnitude, 1u},
      OpArity{rund::kernel::IrOp::Sign, 1u},
      OpArity{rund::kernel::IrOp::AddSat, 2u},
      OpArity{rund::kernel::IrOp::SubSat, 2u},
      OpArity{rund::kernel::IrOp::ShrArithmeticConst, 1u},
  };
  constexpr std::array unsigned_ops{
      OpArity{rund::kernel::IrOp::AddSatUnsigned, 2u},
      OpArity{rund::kernel::IrOp::MinUnsigned, 2u},
      OpArity{rund::kernel::IrOp::MaxUnsigned, 2u},
      OpArity{rund::kernel::IrOp::ClampUnsigned, 3u},
      OpArity{rund::kernel::IrOp::LtUnsigned, 2u},
      OpArity{rund::kernel::IrOp::LeUnsigned, 2u},
      OpArity{rund::kernel::IrOp::GtUnsigned, 2u},
      OpArity{rund::kernel::IrOp::GeUnsigned, 2u},
  };
  constexpr std::array canonical_order{
      OpArity{rund::kernel::IrOp::Min, 2u},
      OpArity{rund::kernel::IrOp::Max, 2u},
      OpArity{rund::kernel::IrOp::Clamp, 3u},
      OpArity{rund::kernel::IrOp::Lt, 2u},
      OpArity{rund::kernel::IrOp::Le, 2u},
      OpArity{rund::kernel::IrOp::Gt, 2u},
      OpArity{rund::kernel::IrOp::Ge, 2u},
  };
  constexpr std::array all_domain{
      OpArity{rund::kernel::IrOp::Neg, 1u},
      OpArity{rund::kernel::IrOp::ShlConst, 1u},
      OpArity{rund::kernel::IrOp::ShrLogicalConst, 1u},
  };

  for (const auto mode : kIntegerDomains) {
    for (const auto spec : signed_ops) {
      const auto ir = IntegerIr(spec.op, spec.arity, mode);
      TEST_ASSERT(Signed(mode.domain) ? rejection_support::Accepts(ir)
                                      : rejection_support::Rejects(
                                            ir, "compute_ir_node_invalid"));
    }
    for (const auto spec : unsigned_ops) {
      const auto ir = IntegerIr(spec.op, spec.arity, mode);
      TEST_ASSERT(Signed(mode.domain) ? rejection_support::Rejects(
                                            ir, "compute_ir_node_invalid")
                                      : rejection_support::Accepts(ir));
    }
    for (const auto spec : canonical_order) {
      const auto ir = IntegerIr(spec.op, spec.arity, mode);
      TEST_ASSERT(Signed(mode.domain) ? rejection_support::Accepts(ir)
                                      : rejection_support::Rejects(
                                            ir, "compute_ir_node_invalid"));
    }
    for (const auto spec : all_domain) {
      TEST_ASSERT(
          rejection_support::Accepts(IntegerIr(spec.op, spec.arity, mode)));
    }
  }
  return 0;
}

} // namespace

int Domain() {
  if (LargeChain() != 0 || FixedOnly() != 0 || Divide() != 0) {
    return 1;
  }
  return Signedness();
}

} // namespace program_compute_contract::lowering_reject
