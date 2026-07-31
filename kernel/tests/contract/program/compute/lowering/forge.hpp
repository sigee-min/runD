#pragma once

#include "contract/program/compute/lowering/bytes.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline std::vector<rund::kernel::u8> ConstShiftIrBytes(
    const rund::kernel::ComputeScalar scalar,
    const rund::kernel::IrOp op,
    const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs,
    const rund::kernel::u32 aux) {
  const rund::kernel::u32 element_bytes =
      scalar == rund::kernel::ComputeScalar::Lane64 ? 8u : 4u;
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "forged-fixed-bit-shift");
  AppendU8(bytes, scalar == rund::kernel::ComputeScalar::Lane64
                      ? kI64NumericMode
                      : kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  AppendBinding(bytes, 2u, "lhs", element_bytes);
  AppendBinding(bytes, 2u, "rhs", element_bytes);
  AppendBinding(bytes, 3u, "out", element_bytes);

  AppendU32(bytes, 4u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, op, lhs, rhs, aux);
  AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> FixedArithmeticBinaryIrBytes(
    const rund::kernel::IrOp op,
    const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs,
    const rund::kernel::u32 aux) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "forged-fixed-arithmetic-binary");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 4u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, op, lhs, rhs, aux);
  AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> FixedArithmeticTernaryIrBytes(
    const rund::kernel::IrOp op,
    const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs,
    const rund::kernel::u32 aux) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "forged-fixed-arithmetic-ternary");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 4u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 2u, "addend");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 5u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 2u);
  AppendNode(bytes, op, lhs, rhs, aux);
  AppendNode(bytes, rund::kernel::IrOp::Write, 4u, 0u, 3u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> ForgedReadPayloadIrBytes() {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "bad-read-payload");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 2u);

  AppendU8(bytes, 2u);
  AppendU8(bytes, kI32NumericMode);
  AppendBytes(bytes, "input");
  AppendU32(bytes, 4u);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 1u);
  AppendU8(bytes, 0xabu);

  AppendU8(bytes, 3u);
  AppendU8(bytes, kI32NumericMode);
  AppendBytes(bytes, "out");
  AppendU32(bytes, 4u);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 0u);

  AppendU32(bytes, 2u);
  AppendU8(bytes,
           static_cast<rund::kernel::u8>(rund::kernel::IrOp::Read));
  AppendU32(bytes, 0u);
  AppendU32(bytes, 0u);
  AppendU32(bytes, 0u);
  AppendU8(bytes,
           static_cast<rund::kernel::u8>(rund::kernel::IrOp::Write));
  AppendU32(bytes, 1u);
  AppendU32(bytes, 0u);
  AppendU32(bytes, 1u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8>
ForgedDuplicateSameKindBindingIrBytes(const rund::kernel::u8 kind) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes,
              kind == 2u ? "duplicate-read-binding"
                         : "duplicate-write-binding");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  if (kind == 2u) {
    AppendBinding(bytes, 2u, "same");
    AppendBinding(bytes, 2u, "same");
    AppendBinding(bytes, 3u, "out");

    AppendU32(bytes, 4u);
    AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
    AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
    AppendNode(bytes, rund::kernel::IrOp::Add, 1u, 2u, 0u);
    AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
    return bytes;
  }

  AppendBinding(bytes, 2u, "input");
  AppendBinding(bytes, 3u, "same");
  AppendBinding(bytes, 3u, "same");

  AppendU32(bytes, 3u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 1u, 0u, 1u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 1u, 0u, 2u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8>
ForgedReadWriteSameNameIrBytes() {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "read-write-same-name");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 2u);

  AppendBinding(bytes, 2u, "same");
  AppendBinding(bytes, 3u, "same");

  AppendU32(bytes, 2u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 1u, 0u, 1u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> UnsupportedBinaryOpIrBytes(
    const rund::kernel::IrOp op) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "unsupported-binary");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 4u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, op, 1u, 2u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> UnsupportedTernaryOpIrBytes(
    const rund::kernel::IrOp op) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "unsupported-ternary");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 4u);

  AppendParamBinding(bytes, "param", 7u);
  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 5u);
  AppendNode(bytes, rund::kernel::IrOp::Param, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 2u);
  AppendNode(bytes, op, 1u, 2u, 3u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 4u, 0u, 3u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> WrongUnaryArityIrBytes(
    const rund::kernel::IrOp op) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "wrong-unary-arity");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 4u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, op, 1u, 2u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
  return bytes;
}

[[nodiscard]] inline std::vector<rund::kernel::u8> WrongBinaryAuxIrBytes(
    const rund::kernel::IrOp op) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "wrong-binary-aux");
  AppendU8(bytes, kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 4u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 2u, "aux");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 5u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 2u);
  AppendNode(bytes, op, 1u, 2u, 3u);
  AppendNode(bytes, rund::kernel::IrOp::Write, 4u, 0u, 3u);
  return bytes;
}

}  // namespace program_compute_contract::lowering_support
