#include "local.hpp"

#include "backing.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstdint>
#include <memory>

namespace rund_node_test_virtual::product {
namespace {

[[nodiscard]] bool same_admission(
    const rund::compute::DevicePipelineMemoryReport &left,
    const rund::compute::DevicePipelineMemoryReport &right) noexcept {
  return left.backend == right.backend &&
         left.capacity_bytes == right.capacity_bytes &&
         left.committed_bytes == right.committed_bytes &&
         left.preparing_bytes == right.preparing_bytes &&
         left.available_bytes == right.available_bytes &&
         left.peak_committed_bytes == right.peak_committed_bytes &&
         left.peak_preparing_bytes == right.peak_preparing_bytes &&
         left.peak_used_bytes == right.peak_used_bytes &&
         left.admission_count == right.admission_count &&
         left.commit_count == right.commit_count &&
         left.release_count == right.release_count &&
         left.rejection_count == right.rejection_count;
}

[[nodiscard]] bool untouched(const BackingFacts facts) noexcept {
  return facts.read_count == 0u && facts.read_bytes == 0u &&
         facts.write_count == 0u && facts.write_bytes == 0u &&
         facts.observation_count == 0u && facts.observation_bytes == 0u &&
         facts.read_failure_count == 0u && facts.write_failure_count == 0u &&
         facts.partial_write_bytes == 0u;
}

} // namespace

ProductCapabilityResult
CheckProductCapability(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return {.detail = 1};
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("virtual-product-capability", PageElements,
                             [](auto value) { return value + 1; })
          .compile();
  if (!program) {
    return {.detail = 2};
  }
  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  if (!input || !output) {
    return {.detail = 3};
  }

  const MemoryStats memory_before = device.memory();
  const DevicePipelineMemoryReport admission_before = device.pipeline_memory();
  auto prepared = virtual_pipeline(*program, *input, *output,
                                   ResidencyConfig{.slots = SlotCapacity});
  if (prepared) {
    return {.disposition = ProductCapability::Executable};
  }
  if (backend != Backend::Vulkan ||
      prepared.reason() != Reason::BackendUnsupported) {
    return {.detail = 4};
  }

  const MemoryStats memory_after = device.memory();
  const DevicePipelineMemoryReport admission_after = device.pipeline_memory();
  if (memory_after != memory_before ||
      !same_admission(admission_before, admission_after) ||
      !untouched(input_backing->facts()) ||
      !untouched(output_backing->facts()) || !input_backing->tail_poisoned() ||
      !output_backing->tail_poisoned()) {
    return {.detail = 5};
  }
  return {.disposition = ProductCapability::BackendBlocked};
}

} // namespace rund_node_test_virtual::product
