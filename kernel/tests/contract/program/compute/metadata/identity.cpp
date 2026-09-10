#include "contract/program/compute/lowering/support.hpp"
#include "local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>

namespace program_compute_metadata_contract {
namespace {

using namespace program_compute_contract::lowering_support;

int CheckRequiredInputCounts() {
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

int CheckUniformReadIdentity() {
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

} // namespace

int CheckInputIdentity() {
  if (const int result = CheckRequiredInputCounts(); result != 0) {
    return result;
  }
  return CheckUniformReadIdentity();
}

} // namespace program_compute_metadata_contract
