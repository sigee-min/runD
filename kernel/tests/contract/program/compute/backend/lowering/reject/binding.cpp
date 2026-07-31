#include "contract/program/compute/backend/lowering/reject/model.hpp"
#include "test/assert.hpp"

#include <array>
#include <string_view>

namespace program_compute_contract::lowering_reject {
namespace {

using namespace backend_lowering_support;

template <Mode Header, std::size_t Count>
[[nodiscard]] bool ForgedDomains(const std::array<Mode, Count> &modes) {
  using rund::compute_dsl::detail::BindingKind;
  for (const BindingKind role :
       {BindingKind::Param, BindingKind::Read, BindingKind::Write}) {
    const BindingKind source =
        role == BindingKind::Param ? BindingKind::Param : BindingKind::Read;
    const auto valid = ValidBindingIr<Header>(source);
    if (!valid.ok) {
      return false;
    }
    const rund::kernel::u32 target = role == BindingKind::Write ? 1u : 0u;
    for (const Mode forged_mode : modes) {
      auto forged = valid;
      if (!SetMode(forged.canonical_bytes, target,
                   static_cast<rund::kernel::u8>(forged_mode))) {
        return false;
      }
      forged = RehashIr(std::move(forged));
      const auto parsed =
          rund::kernel::compute_lowering_detail::ParseComputeIR(forged);
      if (!parsed.ok || !rejection_support::Rejects(
                            forged, "compute_ir_binding_scalar_mismatch")) {
        return false;
      }
    }
  }
  return true;
}

int Cartesian() {
  TEST_ASSERT(
      (ForgedDomains<Mode::FixedLane32>(std::array{Mode::I32, Mode::U32})));
  TEST_ASSERT(
      (ForgedDomains<Mode::FixedLane64>(std::array{Mode::I64, Mode::U64})));
  TEST_ASSERT((ForgedDomains<Mode::I32>(std::array{Mode::FixedLane32})));
  TEST_ASSERT((ForgedDomains<Mode::U32>(std::array{Mode::FixedLane32})));
  TEST_ASSERT((ForgedDomains<Mode::I64>(std::array{Mode::FixedLane64})));
  TEST_ASSERT((ForgedDomains<Mode::U64>(std::array{Mode::FixedLane64})));
  return 0;
}

int Metadata() {
  const auto integer_op = BuildI32DivideOp();
  TEST_ASSERT(integer_op.ok());
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(integer_op.ir());
  TEST_ASSERT(parsed.ok);
  const rund::kernel::ComputeFixedFormat meaningless{
      .integer_bits = 16u,
      .fraction_bits = 16u,
      .rounding = rund::kernel::ComputeRounding::NearestEven,
      .overflow = rund::kernel::ComputeOverflow::Saturate,
      .approximation = rund::kernel::ComputeApproximation::Exact,
  };

  auto forged_top = parsed;
  forged_top.fixed_format = meaningless;
  TEST_ASSERT(std::string_view{
                  rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                      forged_top, integer_op.ir().scalar)} ==
              "compute_ir_numeric_policy_mismatch");

  auto forged_node = parsed;
  forged_node.nodes.front().fixed_format = meaningless;
  TEST_ASSERT(std::string_view{
                  rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                      forged_node, integer_op.ir().scalar)} ==
              "compute_ir_numeric_policy_mismatch");

  auto serialized_top = integer_op.ir();
  serialized_top.fixed_format = meaningless;
  TEST_ASSERT(
      SetTopLevelFixedFormat(serialized_top.canonical_bytes, meaningless));
  serialized_top = RehashIr(std::move(serialized_top));
  const auto top_artifact = rund::kernel::LowerComputeIR(
      serialized_top, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!top_artifact.ok);
  TEST_ASSERT(std::string_view{top_artifact.reason} ==
              "compute_ir_numeric_policy_mismatch");

  auto serialized_node = integer_op.ir();
  TEST_ASSERT(
      SetNodeFixedFormat(serialized_node.canonical_bytes, 1u, meaningless));
  serialized_node = RehashIr(std::move(serialized_node));
  const auto node_artifact = rund::kernel::LowerComputeIR(
      serialized_node, rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!node_artifact.ok);
  TEST_ASSERT(std::string_view{node_artifact.reason} ==
              "compute_ir_numeric_policy_mismatch");
  return 0;
}

int Width() {
  auto forged = BuildI32DivideOp().ir();
  TEST_ASSERT(SetFirstMode(forged.canonical_bytes,
                           rund::kernel::compute_lowering_detail::DomainModeFor(
                               rund::kernel::ComputeScalar::Lane64,
                               rund::kernel::ComputeDomain::I64)));
  forged = RehashIr(std::move(forged));
  const auto parsed =
      rund::kernel::compute_lowering_detail::ParseComputeIR(forged);
  TEST_ASSERT(parsed.ok);
  TEST_ASSERT(
      rejection_support::Rejects(forged, "compute_ir_binding_scalar_mismatch"));
  return 0;
}

int Mask() {
  const auto verify = [&](const auto &op) {
    TEST_ASSERT(op.ok());
    const auto parsed =
        rund::kernel::compute_lowering_detail::ParseComputeIR(op.ir());
    TEST_ASSERT(parsed.ok);

    const auto read = FindNode(parsed, rund::kernel::IrOp::Read);
    const auto select_node = FindNode(parsed, rund::kernel::IrOp::Select);
    const auto write_node = FindNode(parsed, rund::kernel::IrOp::Write);
    TEST_ASSERT(read != 0u && select_node != 0u && write_node != 0u);
    const auto &select = parsed.nodes[select_node - 1u];
    TEST_ASSERT(select.rhs != 0u && select.aux != 0u);
    TEST_ASSERT(rejection_support::Accepts(op.ir()));

    auto direct_write = parsed;
    direct_write.nodes[write_node - 1u].lhs = read;
    TEST_ASSERT(std::string_view{
                    rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                        direct_write, op.ir().scalar)} ==
                "compute_ir_binding_scalar_mismatch");

    auto noncanonical_constant = parsed;
    noncanonical_constant.nodes[select.rhs - 1u].lhs = 2u;
    TEST_ASSERT(std::string_view{
                    rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                        noncanonical_constant, op.ir().scalar)} ==
                "compute_ir_binding_scalar_mismatch");

    auto signed_output = parsed;
    const auto &write = parsed.nodes[write_node - 1u];
    TEST_ASSERT(write.aux < signed_output.bindings.size());
    signed_output.bindings[write.aux].numeric_mode =
        signed_output.bindings[write.aux].element_bytes ==
                sizeof(rund::kernel::u32)
            ? rund::kernel::compute_lowering_detail::DomainModeFor(
                  rund::kernel::ComputeScalar::Lane32,
                  rund::kernel::ComputeDomain::I32)
            : rund::kernel::compute_lowering_detail::DomainModeFor(
                  rund::kernel::ComputeScalar::Lane64,
                  rund::kernel::ComputeDomain::I64);
    TEST_ASSERT(std::string_view{
                    rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                        signed_output, op.ir().scalar)} ==
                "compute_ir_binding_scalar_mismatch");

    auto serialized_direct = op.ir();
    TEST_ASSERT(SetLastNodeLhs(serialized_direct.canonical_bytes, read));
    serialized_direct = RehashIr(std::move(serialized_direct));
    TEST_ASSERT(rejection_support::Rejects(
        serialized_direct, "compute_ir_binding_scalar_mismatch"));
  };

  const auto narrow = BuildU64MaskOp();
  const auto widen = BuildU32MaskOp();
  verify(narrow);
  verify(widen);

  auto multiple_writes =
      rund::kernel::compute_lowering_detail::ParseComputeIR(narrow.ir());
  TEST_ASSERT(multiple_writes.ok);
  const auto original_write = multiple_writes.nodes.back();
  TEST_ASSERT(static_cast<rund::kernel::IrOp>(original_write.op) ==
              rund::kernel::IrOp::Write);
  multiple_writes.bindings.push_back(
      rund::kernel::compute_lowering_detail::ParsedBinding{
          .kind = rund::kernel::compute_lowering_detail::kWriteBindingKind,
          .numeric_mode = rund::kernel::compute_lowering_detail::DomainModeFor(
              rund::kernel::ComputeScalar::Lane64,
              rund::kernel::ComputeDomain::U64),
          .name = "same-width-output",
          .element_bytes = sizeof(rund::kernel::u64),
          .floating_point_param = false,
          .value_bytes = {},
      });
  auto second_write = original_write;
  second_write.aux =
      static_cast<rund::kernel::u32>(multiple_writes.bindings.size() - 1u);
  multiple_writes.nodes.push_back(second_write);
  TEST_ASSERT(std::string_view{
                  rund::kernel::compute_lowering_detail::ValidateLowerableIR(
                      multiple_writes, narrow.ir().scalar)} ==
              "compute_ir_binding_scalar_mismatch");
  return 0;
}

} // namespace

int Binding() {
  if (Cartesian() != 0 || Mask() != 0 || Metadata() != 0) {
    return 1;
  }
  return Width();
}

} // namespace program_compute_contract::lowering_reject
