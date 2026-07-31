#include "contract/program/compute/lowering/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/retention.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace program_compute_contract {
namespace {

using namespace lowering_support;

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
  TEST_ASSERT(metadata.read_routes.empty());
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
  metadata.read_count = 3u;
  metadata.direct_read_mask = 0x1u;
  metadata.read_routes.push_back(
      rund::kernel::ReadRoute{.source = 2u, .index = 1u, .count = 7u});

  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 0u, 5u) == 5u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 1u, 5u) == 5u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 2u, 5u) == 7u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 3u, 5u) == 0u);
  TEST_ASSERT(rund::kernel::RequiredInputCount(metadata, 0u, 0u) == 0u);
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
  if (test_compute_execution_metadata_maps_checked_fixed_lane32_ir() != 0) {
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
  if (test_compute_retained_string_storage_distinguishes_inline_and_external() !=
      0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
