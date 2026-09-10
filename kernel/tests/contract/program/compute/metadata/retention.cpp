#include "local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/retention.hpp>

#include <string>
#include <utility>

namespace program_compute_metadata_contract {

int CheckRetention() {
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

} // namespace program_compute_metadata_contract
