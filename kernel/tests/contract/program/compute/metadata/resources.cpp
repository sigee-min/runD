#include "local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/lowering/resource.hpp>

#include <array>

namespace program_compute_metadata_contract {
namespace {

enum class ExpectedResourceFamily : rund::kernel::u8 {
  Source,
  Write,
  Unary,
  Binary,
  ConstShift,
  Ternary,
  Unknown,
};

[[nodiscard]] constexpr ExpectedResourceFamily
ExpectedResourcesFor(const rund::kernel::IrOp op) noexcept {
  using rund::kernel::IrOp;
  switch (op) {
  case IrOp::Param:
  case IrOp::Read:
  case IrOp::Constant:
  case IrOp::Index:
  case IrOp::ReadAt:
  case IrOp::ReadUniform:
    return ExpectedResourceFamily::Source;
  case IrOp::Write:
    return ExpectedResourceFamily::Write;
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::PredicateNot:
  case IrOp::BitNot:
  case IrOp::NegPositiveFixed:
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log:
  case IrOp::Quantize:
    return ExpectedResourceFamily::Unary;
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap:
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::Eq:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Ne:
  case IrOp::Gt:
  case IrOp::Ge:
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor:
  case IrOp::AddSat:
  case IrOp::AddSatUnsigned:
  case IrOp::SubSat:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed:
  case IrOp::DivFixed:
  case IrOp::Atan2:
  case IrOp::DivSigned:
  case IrOp::DivUnsigned:
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
    return ExpectedResourceFamily::Binary;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
    return ExpectedResourceFamily::ConstShift;
  case IrOp::Clamp:
  case IrOp::Select:
  case IrOp::MulAddFixed:
  case IrOp::ClampUnsigned:
    return ExpectedResourceFamily::Ternary;
  }
  return ExpectedResourceFamily::Unknown;
}

[[nodiscard]] constexpr bool ResourcesEqual(
    const rund::kernel::compute_lowering_detail::ParsedNodeResources &actual,
    const rund::kernel::u32 ref_count, const bool produces_value, const bool ok,
    const rund::kernel::u32 first = 0u, const rund::kernel::u32 second = 0u,
    const rund::kernel::u32 third = 0u) noexcept {
  return actual.refs[0] == first && actual.refs[1] == second &&
         actual.refs[2] == third && actual.ref_count == ref_count &&
         actual.produces_value == produces_value && actual.ok == ok;
}

[[nodiscard]] constexpr bool ParsedNodeResourceClassifierContract() noexcept {
  using namespace rund::kernel;
  using namespace rund::kernel::compute_lowering_detail;
  constexpr u32 lhs = 0x11111111u;
  constexpr u32 rhs = 0x22222222u;
  constexpr u32 aux = 0x33333333u;
  for (u32 raw_op = 0u; raw_op < 256u; ++raw_op) {
    const ParsedNode node{
        .op = static_cast<u8>(raw_op), .lhs = lhs, .rhs = rhs, .aux = aux};
    const ExpectedResourceFamily expected =
        ExpectedResourcesFor(static_cast<IrOp>(node.op));
    const ParsedNodeResources actual = ParsedNodeResourcesFor(node);
    if ((OpName(node.op) != nullptr) !=
        (expected != ExpectedResourceFamily::Unknown)) {
      return false;
    }
    switch (expected) {
    case ExpectedResourceFamily::Source:
      if (!ResourcesEqual(actual, 0u, true, true)) {
        return false;
      }
      break;
    case ExpectedResourceFamily::Write:
      if (!ResourcesEqual(actual, 1u, false, true, lhs)) {
        return false;
      }
      break;
    case ExpectedResourceFamily::Unary:
    case ExpectedResourceFamily::ConstShift:
      if (!ResourcesEqual(actual, 1u, true, true, lhs)) {
        return false;
      }
      break;
    case ExpectedResourceFamily::Binary:
      if (!ResourcesEqual(actual, 2u, true, true, lhs, rhs)) {
        return false;
      }
      break;
    case ExpectedResourceFamily::Ternary:
      if (!ResourcesEqual(actual, 3u, true, true, lhs, rhs, aux)) {
        return false;
      }
      break;
    case ExpectedResourceFamily::Unknown:
      if (!ResourcesEqual(actual, 0u, false, false)) {
        return false;
      }
      break;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool ParsedNodeResourceFieldContract() noexcept {
  using namespace rund::kernel;
  using namespace rund::kernel::compute_lowering_detail;

  // ReadAt stores the index binding, element count, and source binding in the
  // three payload fields. None is an SSA value edge.
  const ParsedNodeResources read_at = ParsedNodeResourcesFor(ParsedNode{
      .op = static_cast<u8>(IrOp::ReadAt),
      .lhs = 101u,
      .rhs = 202u,
      .aux = 303u,
  });
  if (!ResourcesEqual(read_at, 0u, true, true)) {
    return false;
  }

  // Every admitted write mode consumes lhs only; rhs is the mode and aux is
  // the destination binding.
  constexpr std::array write_modes{
      IrWriteMode::Value,
      IrWriteMode::CheckedOrdinal,
      IrWriteMode::BoundaryMask,
  };
  for (const IrWriteMode mode : write_modes) {
    const ParsedNodeResources write = ParsedNodeResourcesFor(ParsedNode{
        .op = static_cast<u8>(IrOp::Write),
        .lhs = 404u,
        .rhs = static_cast<u32>(mode),
        .aux = 505u,
    });
    if (!ResourcesEqual(write, 1u, false, true, 404u)) {
      return false;
    }
  }

  // Constant shift counts live in aux, not in the value graph.
  constexpr std::array shifts{
      IrOp::ShlConst,
      IrOp::ShrLogicalConst,
      IrOp::ShrArithmeticConst,
  };
  for (const IrOp op : shifts) {
    const ParsedNodeResources shift = ParsedNodeResourcesFor(ParsedNode{
        .op = static_cast<u8>(op), .lhs = 606u, .rhs = 0u, .aux = 31u});
    if (!ResourcesEqual(shift, 1u, true, true, 606u)) {
      return false;
    }
  }

  const ParsedNodeResources duplicate_binary = ParsedNodeResourcesFor(
      ParsedNode{.op = static_cast<u8>(IrOp::Add), .lhs = 707u, .rhs = 707u});
  if (!ResourcesEqual(duplicate_binary, 2u, true, true, 707u, 707u)) {
    return false;
  }
  const ParsedNodeResources duplicate_ternary =
      ParsedNodeResourcesFor(ParsedNode{.op = static_cast<u8>(IrOp::Select),
                                        .lhs = 808u,
                                        .rhs = 808u,
                                        .aux = 808u});
  if (!ResourcesEqual(duplicate_ternary, 3u, true, true, 808u, 808u, 808u)) {
    return false;
  }

  return ResourcesEqual(ParsedNodeResourcesFor(ParsedNode{.op = 0xffu}), 0u,
                        false, false);
}

static_assert(ParsedNodeResourceClassifierContract());
static_assert(ParsedNodeResourceFieldContract());

} // namespace

int CheckResourceContracts() {
  TEST_ASSERT(ParsedNodeResourceClassifierContract());
  TEST_ASSERT(ParsedNodeResourceFieldContract());
  return 0;
}

} // namespace program_compute_metadata_contract
