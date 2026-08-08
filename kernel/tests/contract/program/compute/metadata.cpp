#include "contract/program/compute/lowering/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/resource.hpp>
#include <kernel/program/compute/retention.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace program_compute_contract {
namespace {

using namespace lowering_support;

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

struct MixedResourceBody final {
  using Mode = rund::compute_dsl::detail::ScalarMode;
  using Access = rund::compute_dsl::detail::BindingKind;
  using Binding = rund::compute_dsl::detail::BindingRuntime;

  [[nodiscard]] static constexpr Mode scalar_mode() noexcept {
    return Mode::I32;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }
  [[nodiscard]] const std::vector<Binding> &bindings() const noexcept {
    return values;
  }

  std::vector<Binding> values{
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::U32,
              .name = "index",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "direct",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "uniform",
              .element_bytes = 4u},
      Binding{.kind = Access::Read,
              .numeric_mode = Mode::I32,
              .name = "indexed",
              .element_bytes = 4u},
      Binding{.kind = Access::Write,
              .numeric_mode = Mode::I32,
              .name = "output",
              .element_bytes = 4u},
  };
};

[[nodiscard]] rund::kernel::ComputeIR BuildMixedResourceIr() {
  using namespace rund::compute_dsl::detail;
  MixedResourceBody body{};
  BuildContext context{body.bindings(), MixedResourceBody::scalar_mode()};
  const Expr direct = DynamicRead(context, 1u);
  const Expr uniform = DynamicUniformRead(context, 2u);
  const Expr indexed = DynamicReadAt(context, 3u, 0u, 4u);
  const Expr first = Binary(rund::kernel::IrOp::Add, direct, uniform);
  const Expr second = Binary(rund::kernel::IrOp::Add, first, indexed);
  DynamicWrite(context, 4u, second);
  return BuildIr("resource-summary", body, context);
}

int test_parsed_node_resources_exhaustively_classify_value_edges() {
  TEST_ASSERT(ParsedNodeResourceClassifierContract());
  TEST_ASSERT(ParsedNodeResourceFieldContract());
  return 0;
}

int test_compute_execution_metadata_maps_checked_fixed_lane32_ir() {
  const auto op = BuildFixedLane32Op(7);
  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(op.ir(),
                                           rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "ok");
  TEST_ASSERT(metadata.map.op_hash_hi == op.ir().op_hash_hi);
  TEST_ASSERT(metadata.map.op_hash_lo == op.ir().op_hash_lo);
  TEST_ASSERT(metadata.map.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(metadata.map.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(metadata.map.input_buffer_count == 2u);
  TEST_ASSERT(metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(metadata.map.param_bytes == 4u);
  TEST_ASSERT(metadata.map.metadata_bytes_per_tile == 4u);
  TEST_ASSERT(metadata.read_count == 2u);
  TEST_ASSERT(metadata.write_count == 1u);
  TEST_ASSERT(metadata.direct_read_mask == 0x3u);
  TEST_ASSERT(metadata.uniform_read_mask == 0u);
  TEST_ASSERT(metadata.read_routes.empty());
  TEST_ASSERT(metadata.resource_summary.ok);
  TEST_ASSERT(metadata.resource_summary.analysis_version ==
              rund::kernel::kComputeResourceAnalysisVersion);
  TEST_ASSERT(metadata.resource_summary.peak_live_words == 16u);
  TEST_ASSERT(metadata.resource_summary.direct_read_count == 2u);
  TEST_ASSERT(metadata.resource_summary.uniform_read_count == 0u);
  TEST_ASSERT(metadata.resource_summary.indexed_read_count == 0u);
  TEST_ASSERT(metadata.resource_summary.write_count == 1u);
  TEST_ASSERT(metadata.binding_accesses.size() == 3u);
  TEST_ASSERT(metadata.binding_accesses[0] ==
              rund::kernel::ComputeBindingAccess::Read);
  TEST_ASSERT(metadata.binding_accesses[1] ==
              rund::kernel::ComputeBindingAccess::Read);
  TEST_ASSERT(metadata.binding_accesses[2] ==
              rund::kernel::ComputeBindingAccess::Write);
  TEST_ASSERT(metadata.binding_names.size() == 3u);
  TEST_ASSERT(metadata.binding_names[0] == "pos");
  TEST_ASSERT(metadata.binding_names[1] == "vel");
  TEST_ASSERT(metadata.binding_names[2] == "out");
  TEST_ASSERT(metadata.input_element_bytes.size() == 2u);
  TEST_ASSERT(metadata.input_element_bytes[0] == 4u);
  TEST_ASSERT(metadata.input_element_bytes[1] == 4u);
  TEST_ASSERT(metadata.param_storage.size() == 4u);
  TEST_ASSERT(metadata.param_storage[0] == 7u);
  TEST_ASSERT(metadata.param_storage[1] == 0u);
  TEST_ASSERT(metadata.param_storage[2] == 0u);
  TEST_ASSERT(metadata.param_storage[3] == 0u);
  return 0;
}

int test_compute_resource_summary_counts_ops_and_exact_live_intervals() {
  const rund::kernel::ComputeIR ir = BuildMixedResourceIr();
  const rund::kernel::ComputeResourceSummary summary =
      rund::kernel::BuildComputeResourceSummary(
          ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(ir.ok);
  TEST_ASSERT(summary.ok);
  TEST_ASSERT(std::string_view{summary.reason} == "ok");
  TEST_ASSERT(summary.analysis_version ==
              rund::kernel::kComputeResourceAnalysisVersion);
  TEST_ASSERT(summary.peak_live_words == 4u);
  TEST_ASSERT(summary.direct_read_count == 1u);
  TEST_ASSERT(summary.uniform_read_count == 1u);
  TEST_ASSERT(summary.indexed_read_count == 1u);
  TEST_ASSERT(summary.write_count == 1u);
  return 0;
}

int test_compute_resource_summary_fails_closed_with_admission_reason() {
  rund::kernel::ComputeIR ir = BuildMixedResourceIr();
  ir.op_hash_hi ^= 1u;

  const rund::kernel::ComputeResourceSummary summary =
      rund::kernel::BuildComputeResourceSummary(
          ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!summary.ok);
  TEST_ASSERT(summary.analysis_version == 0u);
  TEST_ASSERT(summary.peak_live_words == 0u);
  TEST_ASSERT(std::string_view{summary.reason} == "compute_ir_hash_mismatch");
  return 0;
}

int test_compute_execution_metadata_rejects_malformed_ir_reason() {
  rund::kernel::ComputeIR ir = BuildFixedLane32Op(7).ir();
  TEST_ASSERT(SetBindingCount(ir.canonical_bytes, 0xffffffffu));
  ir = RehashIr(std::move(ir));

  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} ==
              "compute_ir_binding_count_invalid");
  return 0;
}

int test_compute_execution_metadata_rejects_unsupported_ir_reason() {
  rund::kernel::ComputeIR ir = BuildFixedLane32Op(7).ir();
  TEST_ASSERT(ReplaceFirstNodeOp(ir.canonical_bytes, 0xffu));
  ir = RehashIr(std::move(ir));

  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "compute_ir_op_unsupported");
  return 0;
}

int test_compute_execution_metadata_rejects_hash_mismatch_reason() {
  rund::kernel::ComputeIR ir = BuildFixedLane32Op(7).ir();
  ir.op_hash_hi ^= 1u;

  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "compute_ir_hash_mismatch");
  return 0;
}

int test_compute_execution_metadata_owns_required_input_counts() {
  rund::kernel::ExecutionMetadata metadata{};
  metadata.read_count = 4u;
  metadata.direct_read_mask = 0x1u;
  metadata.uniform_read_mask = 0x9u;
  metadata.read_routes.push_back(
      rund::kernel::ReadRoute{.source = 2u, .index = 1u, .count = 7u});

  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 0u, 5u) == 5u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 1u, 5u) == 5u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 2u, 5u) == 7u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 3u, 5u) == 1u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 4u, 5u) == 0u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 0u, 0u) == 0u);
  return 0;
}

int test_compute_execution_metadata_owns_uniform_read_identity() {
  const rund::kernel::ComputeIR ir = BuildI32UniformReadIr();
  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(ir.ok);
  TEST_ASSERT(metadata.ok);
  TEST_ASSERT(metadata.read_count == 1u);
  TEST_ASSERT(metadata.direct_read_mask == 0u);
  TEST_ASSERT(metadata.uniform_read_mask == 0x1u);
  TEST_ASSERT(metadata.read_routes.empty());
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 0u, 4096u) == 1u);
  return 0;
}

int test_compute_retained_string_storage_distinguishes_inline_and_external() {
  using namespace rund::kernel;
  using compute_retained_detail::StringExternalStorageBytes;
  using compute_retained_detail::VectorCapacityBytes;

  const std::string empty;
  const std::string short_name{"short"};
  const std::string long_name(257u, 'x');
  std::string reserved;
  reserved.reserve(513u);

  TEST_ASSERT(StringExternalStorageBytes(empty) == 0u);
  TEST_ASSERT(StringExternalStorageBytes(short_name) == 0u);
  TEST_ASSERT(StringExternalStorageBytes(long_name) ==
              long_name.capacity() + 1u);
  TEST_ASSERT(reserved.empty());
  TEST_ASSERT(StringExternalStorageBytes(reserved) == reserved.capacity() + 1u);

  ExecutionMetadata metadata{};
  metadata.param_storage.reserve(7u);
  metadata.input_element_bytes.reserve(2u);
  metadata.output_element_bytes.reserve(3u);
  metadata.binding_accesses.reserve(4u);
  metadata.binding_names.reserve(3u);
  metadata.read_routes.reserve(2u);
  metadata.binding_names.push_back(short_name);
  metadata.binding_names.push_back(long_name);
  metadata.binding_names.push_back(reserved);
  u64 expected = VectorCapacityBytes(metadata.param_storage);
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(metadata.input_element_bytes));
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(metadata.output_element_bytes));
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(metadata.binding_accesses));
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(metadata.binding_names));
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(metadata.read_routes));
  for (const std::string &name : metadata.binding_names) {
    expected = compute_retained_detail::Add(expected,
                                            StringExternalStorageBytes(name));
  }
  TEST_ASSERT(metadata.retained_dynamic_memory_bytes() == expected);

  LoweringArtifact artifact{};
  artifact.metadata = std::move(metadata);
  artifact.source_text = long_name;
  artifact.canonical_ir_bytes.reserve(13u);
  expected = artifact.metadata.retained_dynamic_memory_bytes();
  expected = compute_retained_detail::Add(expected,
                                          artifact.source_text.capacity() + 1u);
  expected = compute_retained_detail::Add(
      expected, VectorCapacityBytes(artifact.canonical_ir_bytes));
  TEST_ASSERT(artifact.retained_dynamic_memory_bytes() == expected);
  return 0;
}

} // namespace

int RunComputeMetadataContract() {
  if (test_parsed_node_resources_exhaustively_classify_value_edges() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_maps_checked_fixed_lane32_ir() != 0) {
    return 1;
  }
  if (test_compute_resource_summary_counts_ops_and_exact_live_intervals() !=
      0) {
    return 1;
  }
  if (test_compute_resource_summary_fails_closed_with_admission_reason() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_rejects_malformed_ir_reason() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_rejects_unsupported_ir_reason() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_rejects_hash_mismatch_reason() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_owns_required_input_counts() != 0) {
    return 1;
  }
  if (test_compute_execution_metadata_owns_uniform_read_identity() != 0) {
    return 1;
  }
  if (test_compute_retained_string_storage_distinguishes_inline_and_external() !=
      0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
