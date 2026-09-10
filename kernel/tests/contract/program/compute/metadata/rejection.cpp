#include "contract/program/compute/lowering/support.hpp"
#include "local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <string_view>
#include <utility>

namespace program_compute_metadata_contract {
namespace {

using namespace program_compute_contract::lowering_support;

int CheckMalformedIr() {
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

int CheckUnsupportedIr() {
  rund::kernel::ComputeIR ir = BuildFixedLane32Op(7).ir();
  TEST_ASSERT(ReplaceFirstNodeOp(ir.canonical_bytes, 0xffu));
  ir = RehashIr(std::move(ir));

  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "compute_ir_op_unsupported");
  return 0;
}

int CheckHashMismatch() {
  rund::kernel::ComputeIR ir = BuildFixedLane32Op(7).ir();
  ir.op_hash_hi ^= 1u;

  const rund::kernel::ExecutionMetadata metadata =
      rund::kernel::BuildExecutionMetadata(ir, rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!metadata.ok);
  TEST_ASSERT(std::string_view{metadata.reason} == "compute_ir_hash_mismatch");
  return 0;
}

} // namespace

int CheckMetadataRejections() {
  if (const int result = CheckMalformedIr(); result != 0) {
    return result;
  }
  if (const int result = CheckUnsupportedIr(); result != 0) {
    return result;
  }
  return CheckHashMismatch();
}

} // namespace program_compute_metadata_contract
