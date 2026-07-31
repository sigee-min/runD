#include "contract/program/compute/backend/lowering/reject/local.hpp"
#include "test/assert.hpp"

#include <cstdio>
#include <string_view>

namespace program_compute_contract::lowering_reject {
namespace {

using namespace backend_lowering_support;

int UnsupportedOp() {
  auto ir = BuildFixedLane32Op(7).ir();
  TEST_ASSERT(ReplaceFirstNodeOp(ir.canonical_bytes, 0xffu));
  ir = RehashIr(std::move(ir));
  const auto artifact =
      rund::kernel::LowerComputeIR(ir, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!artifact.ok);
  TEST_ASSERT(std::string_view{artifact.reason} == "compute_ir_op_unsupported");
  TEST_ASSERT(artifact.source_text.empty());
  return 0;
}

int Arity() {
  const auto unary = rund::kernel::LowerComputeIR(
      IrFromBytes(WrongUnaryArityIrBytes(rund::kernel::IrOp::Abs)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!unary.ok);
  TEST_ASSERT(std::string_view{unary.reason} == "compute_ir_node_invalid");
  TEST_ASSERT(unary.source_text.empty());

  const auto binary = rund::kernel::LowerComputeIR(
      IrFromBytes(WrongBinaryAuxIrBytes(rund::kernel::IrOp::Ne)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!binary.ok);
  TEST_ASSERT(std::string_view{binary.reason} == "compute_ir_node_invalid");
  TEST_ASSERT(binary.source_text.empty());
  return 0;
}

int Shift() {
  const auto wrong_arity = rund::kernel::LowerComputeIR(
      IrFromBytes(ConstShiftIrBytes(rund::kernel::ComputeScalar::Lane32,
                                    rund::kernel::IrOp::ShlConst, 1u, 2u, 3u)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!wrong_arity.ok);
  TEST_ASSERT(std::string_view{wrong_arity.reason} ==
              "compute_ir_node_invalid");
  TEST_ASSERT(wrong_arity.source_text.empty());

  const auto bad_ref = rund::kernel::LowerComputeIR(
      IrFromBytes(ConstShiftIrBytes(rund::kernel::ComputeScalar::Lane32,
                                    rund::kernel::IrOp::ShrLogicalConst, 9u, 0u,
                                    3u)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!bad_ref.ok);
  TEST_ASSERT(std::string_view{bad_ref.reason} == "compute_ir_node_invalid");
  TEST_ASSERT(bad_ref.source_text.empty());

  const auto overshift32 = rund::kernel::LowerComputeIR(
      IrFromBytes(ConstShiftIrBytes(rund::kernel::ComputeScalar::Lane32,
                                    rund::kernel::IrOp::ShrArithmeticConst, 1u,
                                    0u, 32u)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!overshift32.ok);
  TEST_ASSERT(std::string_view{overshift32.reason} ==
              "compute_shift_count_invalid");
  TEST_ASSERT(overshift32.source_text.empty());

  const auto overshift64 = rund::kernel::LowerComputeIR(
      IrFromBytes(ConstShiftIrBytes(rund::kernel::ComputeScalar::Lane64,
                                    rund::kernel::IrOp::ShrArithmeticConst, 1u,
                                    0u, 64u),
                  rund::kernel::ComputeScalar::Lane64),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!overshift64.ok);
  TEST_ASSERT(std::string_view{overshift64.reason} ==
              "compute_shift_count_invalid");
  TEST_ASSERT(overshift64.source_text.empty());
  return 0;
}

int Nonlinear() {
  const auto unary =
      rund::kernel::LowerComputeIR(IrFromBytes(FixedNonlinearUnaryIrBytes(
                                       rund::kernel::IrOp::Sqrt, 1u, 2u, 0u)),
                                   rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!unary.ok);
  TEST_ASSERT(std::string_view{unary.reason} == "compute_ir_node_invalid");
  TEST_ASSERT(unary.source_text.empty());

  const auto binary = rund::kernel::LowerComputeIR(
      IrFromBytes(WrongBinaryAuxIrBytes(rund::kernel::IrOp::DivFixed)),
      rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!binary.ok);
  TEST_ASSERT(std::string_view{binary.reason} == "compute_ir_node_invalid");
  TEST_ASSERT(binary.source_text.empty());
  return 0;
}

int Payload() {
  const auto op = BuildFixedLane32Op(7);
  auto huge_bindings = op.ir();
  TEST_ASSERT(SetBindingCount(huge_bindings.canonical_bytes, 0xffffffffu));
  huge_bindings = RehashIr(std::move(huge_bindings));
  const auto huge = rund::kernel::LowerComputeIR(
      huge_bindings, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!huge.ok);
  TEST_ASSERT(std::string_view{huge.reason} ==
              "compute_ir_binding_count_invalid");

  auto wrong_param_size = op.ir();
  TEST_ASSERT(
      SetFirstBindingElementBytes(wrong_param_size.canonical_bytes, 8u));
  wrong_param_size = RehashIr(std::move(wrong_param_size));
  const auto param = rund::kernel::LowerComputeIR(
      wrong_param_size, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!param.ok);
  TEST_ASSERT(std::string_view{param.reason} ==
              "compute_ir_param_size_mismatch");

  const auto read_payload = rund::kernel::LowerComputeIR(
      IrFromBytes(ForgedReadPayloadIrBytes()), rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!read_payload.ok);
  TEST_ASSERT(std::string_view{read_payload.reason} ==
              "compute_ir_binding_payload_invalid");
  return 0;
}

int Quantize() {
  auto ir = BuildFixedLane32Op(7).ir();
  const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
  TEST_ASSERT(parsed.ok && !parsed.nodes.empty());
  const auto &write = parsed.nodes.back();
  TEST_ASSERT(static_cast<rund::kernel::IrOp>(write.op) ==
              rund::kernel::IrOp::Write);
  TEST_ASSERT(write.lhs > 0u && write.lhs <= parsed.nodes.size());
  const auto &quantize = parsed.nodes[write.lhs - 1u];
  TEST_ASSERT(static_cast<rund::kernel::IrOp>(quantize.op) ==
              rund::kernel::IrOp::Quantize);
  TEST_ASSERT(SetLastNodeLhs(ir.canonical_bytes, quantize.lhs));
  ir = RehashIr(std::move(ir));

  const auto artifact =
      rund::kernel::LowerComputeIR(ir, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!artifact.ok);
  if (std::string_view{artifact.reason} != "compute_ir_quantize_required") {
    std::fprintf(stderr, "unexpected write-boundary reason: %s\n",
                 artifact.reason);
  }
  TEST_ASSERT(std::string_view{artifact.reason} ==
              "compute_ir_quantize_required");
  TEST_ASSERT(artifact.source_text.empty());
  return 0;
}

int Approximation() {
  auto ir = BuildFixedLane32NonlinearOps().ir();
  const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
  TEST_ASSERT(parsed.ok);
  rund::kernel::u32 quantize = 0u;
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    if (static_cast<rund::kernel::IrOp>(parsed.nodes[index].op) ==
        rund::kernel::IrOp::Quantize) {
      quantize = static_cast<rund::kernel::u32>(index + 1u);
    }
  }
  TEST_ASSERT(quantize != 0u);
  TEST_ASSERT(SetNodeApproximation(ir.canonical_bytes, quantize,
                                   rund::kernel::ComputeApproximation::Exact));
  ir = RehashIr(std::move(ir));

  const auto artifact =
      rund::kernel::LowerComputeIR(ir, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!artifact.ok);
  TEST_ASSERT(std::string_view{artifact.reason} ==
              "compute_ir_approximation_downgrade");
  TEST_ASSERT(artifact.source_text.empty());
  return 0;
}

} // namespace

int Shape() {
  if (UnsupportedOp() != 0 || Arity() != 0 || Shift() != 0 ||
      Nonlinear() != 0 || Quantize() != 0 || Approximation() != 0) {
    return 1;
  }
  return Payload();
}

} // namespace program_compute_contract::lowering_reject
