#include "contract/program/compute/backend/lowering/reject/model.hpp"
#include "test/assert.hpp"

#include <array>

namespace program_compute_contract::lowering_reject {
namespace {

using namespace backend_lowering_support;

int Modes() {
  TEST_ASSERT((CarrierRejects<Mode::I32, Mode::U32>()));
  TEST_ASSERT((CarrierRejects<Mode::U32, Mode::I32>()));
  TEST_ASSERT((CarrierRejects<Mode::I64, Mode::U64>()));
  TEST_ASSERT((CarrierRejects<Mode::U64, Mode::I64>()));
  TEST_ASSERT((BoundaryRejects<Mode::U32, Mode::I32>()));
  TEST_ASSERT((BoundaryRejects<Mode::U32, Mode::FixedLane32>()));
  TEST_ASSERT((BoundaryRejects<Mode::U64, Mode::FixedLane64>()));
  TEST_ASSERT(
      (FormatRejects<Mode::I32, Mode::U32>(rejection_support::kFixed16x16)));
  TEST_ASSERT(
      (FormatRejects<Mode::I64, Mode::U64>(rejection_support::kFixed20x44)));

  const std::array checked{
      CheckedOrdinalIr<Mode::I32, Mode::U32>(),
      CheckedOrdinalIr<Mode::U32, Mode::I32>(),
      CheckedOrdinalIr<Mode::I64, Mode::U64>(),
      CheckedOrdinalIr<Mode::U64, Mode::I64>(),
  };
  for (const auto &ir : checked) {
    TEST_ASSERT(ir.ok);
    TEST_ASSERT(rejection_support::Accepts(ir));
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
    TEST_ASSERT(parsed.ok);
    const auto read = FindNode(parsed, rund::kernel::IrOp::Read);
    const auto select = FindNode(parsed, rund::kernel::IrOp::Select);
    const auto write = FindNode(parsed, rund::kernel::IrOp::Write);
    TEST_ASSERT(read != 0u && select != 0u && write != 0u);
    TEST_ASSERT(parsed.nodes[write - 1u].rhs ==
                static_cast<rund::kernel::u32>(
                    rund::kernel::IrWriteMode::CheckedOrdinal));

    auto ordinary = ir;
    TEST_ASSERT(SetNode(
        ordinary.canonical_bytes, write, rund::kernel::IrOp::Write,
        parsed.nodes[write - 1u].lhs,
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::Value),
        parsed.nodes[write - 1u].aux));
    ordinary = RehashIr(std::move(ordinary));
    TEST_ASSERT(ordinary.op_hash_hi != ir.op_hash_hi ||
                ordinary.op_hash_lo != ir.op_hash_lo);
    TEST_ASSERT(
        rejection_support::Rejects(ordinary, "compute_ir_node_invalid"));

    auto mismatched_source = ir;
    TEST_ASSERT(
        SetNode(mismatched_source.canonical_bytes, select,
                rund::kernel::IrOp::Select, parsed.nodes[select - 1u].lhs,
                parsed.nodes[select - 1u].aux, parsed.nodes[select - 1u].aux));
    mismatched_source = RehashIr(std::move(mismatched_source));
    TEST_ASSERT(rejection_support::Rejects(mismatched_source,
                                           "compute_ir_node_invalid"));

    auto wrong_mode = ir;
    TEST_ASSERT(SetNode(
        wrong_mode.canonical_bytes, write, rund::kernel::IrOp::Write,
        parsed.nodes[write - 1u].lhs,
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::BoundaryMask),
        parsed.nodes[write - 1u].aux));
    wrong_mode = RehashIr(std::move(wrong_mode));
    TEST_ASSERT(
        rejection_support::Rejects(wrong_mode, "compute_ir_node_invalid"));
  }

  const std::array unsigned_boundary{
      BoundaryIr<Mode::I32, Mode::U32>({}),
      BoundaryIr<Mode::I64, Mode::U64>({}),
      BoundaryIr<Mode::I64, Mode::U32>({}),
      BoundaryIr<Mode::I32, Mode::U64>({}),
  };
  const std::array public_mask{
      ValueMaskIr<Mode::I32, Mode::U32>(),
      ValueMaskIr<Mode::I64, Mode::U64>(),
      ValueMaskIr<Mode::I64, Mode::U32>(),
      ValueMaskIr<Mode::I32, Mode::U64>(),
  };
  for (std::size_t index = 0u; index < unsigned_boundary.size(); ++index) {
    const auto &ir = unsigned_boundary[index];
    const auto &public_ir = public_mask[index];
    TEST_ASSERT(ir.ok && public_ir.ok);
    TEST_ASSERT(ir.canonical_bytes == public_ir.canonical_bytes);
    TEST_ASSERT(ir.op_hash_hi == public_ir.op_hash_hi);
    TEST_ASSERT(ir.op_hash_lo == public_ir.op_hash_lo);
    TEST_ASSERT(rejection_support::Accepts(ir));
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
    TEST_ASSERT(parsed.ok);
    const auto write = FindNode(parsed, rund::kernel::IrOp::Write);
    TEST_ASSERT(write != 0u);
    TEST_ASSERT(
        parsed.nodes[write - 1u].rhs ==
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::Value));
    TEST_ASSERT(rund::kernel::ComputeFixedFormatAbsent(
        parsed.nodes[write - 1u].fixed_format));
    TEST_ASSERT(static_cast<rund::kernel::IrOp>(
                    parsed.nodes[parsed.nodes[write - 1u].lhs - 1u].op) ==
                rund::kernel::IrOp::Select);
    auto duplicate_mode = ir;
    TEST_ASSERT(SetNode(
        duplicate_mode.canonical_bytes, write, rund::kernel::IrOp::Write,
        parsed.nodes[write - 1u].lhs,
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::BoundaryMask),
        parsed.nodes[write - 1u].aux));
    duplicate_mode = RehashIr(std::move(duplicate_mode));
    TEST_ASSERT(rejection_support::Rejects(
        duplicate_mode, index < 2u ? "compute_ir_node_invalid"
                                   : "compute_ir_binding_scalar_mismatch"));
  }

  const std::array boundary{
      BoundaryIr<Mode::U32, Mode::I32>({}),
      BoundaryIr<Mode::I32, Mode::FixedLane32>(rejection_support::kFixed16x16),
      BoundaryIr<Mode::U32, Mode::FixedLane32>(rejection_support::kFixed16x16),
      BoundaryIr<Mode::U64, Mode::I64>({}),
      BoundaryIr<Mode::I64, Mode::FixedLane64>(rejection_support::kFixed20x44),
      BoundaryIr<Mode::U64, Mode::FixedLane64>(rejection_support::kFixed20x44),
  };
  for (const auto &ir : boundary) {
    TEST_ASSERT(ir.ok);
    TEST_ASSERT(rejection_support::Accepts(ir));
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
    TEST_ASSERT(parsed.ok);
    const auto read = FindNode(parsed, rund::kernel::IrOp::Read);
    const auto predicate = FindNode(parsed, rund::kernel::IrOp::Ne);
    const auto select = FindNode(parsed, rund::kernel::IrOp::Select);
    const auto write = FindNode(parsed, rund::kernel::IrOp::Write);
    TEST_ASSERT(read != 0u && predicate != 0u && select != 0u && write != 0u);
    TEST_ASSERT(parsed.nodes[write - 1u].rhs ==
                static_cast<rund::kernel::u32>(
                    rund::kernel::IrWriteMode::BoundaryMask));
    TEST_ASSERT(parsed.nodes[write - 1u].lhs == select);
    TEST_ASSERT(parsed.nodes[select - 1u].lhs == predicate);

    auto ordinary = ir;
    TEST_ASSERT(SetNode(
        ordinary.canonical_bytes, write, rund::kernel::IrOp::Write,
        parsed.nodes[write - 1u].lhs,
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::Value),
        parsed.nodes[write - 1u].aux));
    ordinary = RehashIr(std::move(ordinary));
    TEST_ASSERT(ordinary.op_hash_hi != ir.op_hash_hi ||
                ordinary.op_hash_lo != ir.op_hash_lo);
    const auto target_domain =
        rund::kernel::compute_lowering_detail::BindingDomainForShape(
            parsed.bindings[parsed.nodes[write - 1u].aux]);
    TEST_ASSERT(rejection_support::Rejects(
        ordinary, target_domain == rund::kernel::ComputeDomain::Fixed
                      ? "compute_ir_numeric_policy_mismatch"
                      : "compute_ir_node_invalid"));

    auto direct = ir;
    TEST_ASSERT(SetNode(
        direct.canonical_bytes, write, rund::kernel::IrOp::Write, read,
        static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::BoundaryMask),
        parsed.nodes[write - 1u].aux));
    direct = RehashIr(std::move(direct));
    TEST_ASSERT(rejection_support::Rejects(direct, "compute_ir_node_invalid"));

    auto wrong_predicate = ir;
    TEST_ASSERT(SetNode(wrong_predicate.canonical_bytes, predicate,
                        rund::kernel::IrOp::Eq,
                        parsed.nodes[predicate - 1u].lhs,
                        parsed.nodes[predicate - 1u].rhs, 0u));
    wrong_predicate = RehashIr(std::move(wrong_predicate));
    TEST_ASSERT(
        rejection_support::Rejects(wrong_predicate, "compute_ir_node_invalid"));

    auto wrong_one = ir;
    const auto one = parsed.nodes[select - 1u].rhs;
    TEST_ASSERT(SetNode(wrong_one.canonical_bytes, one,
                        rund::kernel::IrOp::Constant, 2u, 0u, 0u));
    wrong_one = RehashIr(std::move(wrong_one));
    TEST_ASSERT(
        rejection_support::Rejects(wrong_one, "compute_ir_node_invalid"));
  }

  const auto signed_boundary = boundary.front();
  const auto signed_parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(signed_boundary);
  TEST_ASSERT(signed_parsed.ok);
  const auto signed_write = FindNode(signed_parsed, rund::kernel::IrOp::Write);
  TEST_ASSERT(signed_write != 0u);
  auto unknown_mode = signed_boundary;
  TEST_ASSERT(SetNode(unknown_mode.canonical_bytes, signed_write,
                      rund::kernel::IrOp::Write,
                      signed_parsed.nodes[signed_write - 1u].lhs, 3u,
                      signed_parsed.nodes[signed_write - 1u].aux));
  unknown_mode = RehashIr(std::move(unknown_mode));
  TEST_ASSERT(unknown_mode.op_hash_hi != signed_boundary.op_hash_hi ||
              unknown_mode.op_hash_lo != signed_boundary.op_hash_lo);
  TEST_ASSERT(
      rejection_support::Rejects(unknown_mode, "compute_ir_node_invalid"));

  const auto fixed_boundary = boundary[1u];
  const auto fixed_parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed_boundary);
  TEST_ASSERT(fixed_parsed.ok);
  const auto fixed_write = FindNode(fixed_parsed, rund::kernel::IrOp::Write);
  TEST_ASSERT(fixed_write != 0u);
  auto ordinary_fixed = fixed_boundary;
  TEST_ASSERT(SetNode(
      ordinary_fixed.canonical_bytes, fixed_write, rund::kernel::IrOp::Write,
      fixed_parsed.nodes[fixed_write - 1u].lhs,
      static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::Value),
      fixed_parsed.nodes[fixed_write - 1u].aux));
  ordinary_fixed = RehashIr(std::move(ordinary_fixed));
  TEST_ASSERT(rejection_support::Rejects(ordinary_fixed,
                                         "compute_ir_numeric_policy_mismatch"));

  auto wrong_format = fixed_boundary;
  auto invalid_format = rejection_support::kFixed16x16;
  invalid_format.integer_bits = 17u;
  TEST_ASSERT(SetNodeFixedFormat(wrong_format.canonical_bytes, fixed_write,
                                 invalid_format));
  wrong_format = RehashIr(std::move(wrong_format));
  TEST_ASSERT(rejection_support::Rejects(wrong_format,
                                         "compute_ir_numeric_policy_invalid"));

  const std::array expressions{
      CheckedOrdinalIr<Mode::I32, Mode::U32, true>(),
      CheckedOrdinalIr<Mode::U64, Mode::I64, true>(),
      BoundaryIr<Mode::I32, Mode::FixedLane32, true>(
          rejection_support::kFixed16x16),
      BoundaryIr<Mode::U64, Mode::I64, true>({}),
  };
  for (const auto &ir : expressions) {
    TEST_ASSERT(ir.ok);
    TEST_ASSERT(rejection_support::Accepts(ir));
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
    TEST_ASSERT(parsed.ok);
    TEST_ASSERT(FindNode(parsed, rund::kernel::IrOp::Add) != 0u);
  }
  return 0;
}

} // namespace

int Carrier() { return Modes(); }

} // namespace program_compute_contract::lowering_reject
