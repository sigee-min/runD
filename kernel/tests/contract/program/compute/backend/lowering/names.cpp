#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

int test_compute_lowering_mangles_names_and_escapes_identity_text() {
  const auto op = BuildUnsafeNameOp(7);
  const rund::kernel::LoweringArtifact metal =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(metal.ok);
  TEST_ASSERT(vulkan.ok);
  TEST_ASSERT(metal.source_text.find("param_64742d6d73") !=
              std::string_view::npos);
  TEST_ASSERT(metal.source_text.find("read_706f73206261640a78") !=
              std::string_view::npos);
  TEST_ASSERT(metal.source_text.find("write_6f757421") !=
              std::string_view::npos);
  TEST_ASSERT(metal.source_text.find("operation_hex=") !=
              std::string_view::npos);
  TEST_ASSERT(vulkan.source_text.find("name_hex=706f73206261640a78") !=
              std::string_view::npos);
  return 0;
}

int test_compute_lowering_mangling_is_collision_safe() {
  const auto op = BuildCollisionNameOp();
  const rund::kernel::LoweringArtifact metal =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(metal.ok);
  const std::string_view source{metal.source_text};
  TEST_ASSERT(CountOccurrences(source,
                               "const device uchar* read_612d62 [[buffer(") ==
              1u);
  TEST_ASSERT(CountOccurrences(
                  source,
                  "const device uchar* read_615f7832645f62 [[buffer(") == 1u);
  TEST_ASSERT(CountOccurrences(source,
                               "const device uchar* read_31 [[buffer(") == 1u);
  TEST_ASSERT(CountOccurrences(
                  source,
                  "const device uchar* read_72756e645f31 [[buffer(") == 1u);
  TEST_ASSERT(source.find("name_hex=612d62") != std::string_view::npos);
  TEST_ASSERT(source.find("name_hex=615f7832645f62") !=
              std::string_view::npos);
  TEST_ASSERT(source.find("name_hex=31") != std::string_view::npos);
  TEST_ASSERT(source.find("name_hex=72756e645f31") !=
              std::string_view::npos);
  return 0;
}

int test_compute_lowering_rejects_duplicate_same_kind_binding_names() {
  const rund::kernel::LoweringArtifact duplicate_read =
      rund::kernel::LowerComputeIR(
          IrFromBytes(ForgedDuplicateSameKindBindingIrBytes(2u)),
          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!duplicate_read.ok);
  TEST_ASSERT(std::string_view{duplicate_read.reason} ==
              "compute_ir_binding_duplicate");
  TEST_ASSERT(duplicate_read.source_text.empty());

  const rund::kernel::LoweringArtifact duplicate_write =
      rund::kernel::LowerComputeIR(
          IrFromBytes(ForgedDuplicateSameKindBindingIrBytes(3u)),
          rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(!duplicate_write.ok);
  TEST_ASSERT(std::string_view{duplicate_write.reason} ==
              "compute_ir_binding_duplicate");
  TEST_ASSERT(duplicate_write.source_text.empty());

  const rund::kernel::LoweringArtifact read_write =
      rund::kernel::LowerComputeIR(IrFromBytes(ForgedReadWriteSameNameIrBytes()),
                                   rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(read_write.ok);
  const std::string_view source{read_write.source_text};
  TEST_ASSERT(CountOccurrences(source,
                               "const device uchar* read_73616d65 "
                               "[[buffer(") == 1u);
  TEST_ASSERT(CountOccurrences(source,
                               "device uchar* write_73616d65 [[buffer(") ==
              1u);
  return 0;
}

}  // namespace

int RunComputeBackendLoweringNameContract() {
  if (test_compute_lowering_mangles_names_and_escapes_identity_text() != 0) {
    return 1;
  }
  if (test_compute_lowering_mangling_is_collision_safe() != 0) {
    return 1;
  }
  return test_compute_lowering_rejects_duplicate_same_kind_binding_names();
}

}  // namespace program_compute_contract
