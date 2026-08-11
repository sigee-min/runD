#include "local.hpp"

#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

int CheckProductDeviceCache(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-device-cache", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(ElementPageBytes,
                                                              ElementPageBytes);
  auto first_backing = std::make_shared<MemoryVirtualBacking>(ElementPageBytes,
                                                              ElementPageBytes);
  auto second_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  std::array<std::int32_t, PageElements> seeded{};
  SeedInput(seeded);
  if (!input_backing->seed(std::as_bytes(std::span{seeded}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(PageElements, input_backing);
  auto first = virtual_buffer<std::int32_t>(PageElements, first_backing);
  auto second = virtual_buffer<std::int32_t>(PageElements, second_backing);
  const ResidencyConfig config{
      .device_resident_bytes = ElementPageBytes * 4u,
      .host_staging_bytes = ElementPageBytes * 4u,
  };
  auto first_pipeline =
      input && first
          ? virtual_pipeline(*program, *input, *first, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  auto second_pipeline =
      input && second
          ? virtual_pipeline(*program, *input, *second, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!first_pipeline || !second_pipeline || !first_pipeline->run()) {
    return 4;
  }
  const BackingFacts after_first = input_backing->facts();
  if (after_first.read_count != 1u || !second_pipeline->run()) {
    return 5;
  }
  const ResidencyStats &cache = second_pipeline->stats().pipeline.residency;
  std::array<std::int32_t, PageElements> observed{};
  if (!second_backing->observe(std::as_writable_bytes(std::span{observed}))) {
    return 6;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != ProductValue(SeedValue(index))) {
      return 7;
    }
  }
  if (input_backing->facts().read_count != after_first.read_count ||
      cache.page_in_count != 0u || cache.cache_hit_count != 1u ||
      cache.late_page_count != 0u || cache.prefetch_count != 0u ||
      cache.page_out_count != 1u) {
    return 8;
  }

  // A failed terminal write must discard the dirty global frame. Otherwise a
  // later Pipeline with a different input/output backing would write the old
  // bytes to its output before computing its own page.
  auto failed_input_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  auto failed_output_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  auto replacement_input_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  auto replacement_output_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  std::array<std::int32_t, PageElements> replacement_seed{};
  for (std::size_t index = 0u; index < replacement_seed.size(); ++index) {
    replacement_seed[index] = static_cast<std::int32_t>(100u + index);
  }
  if (!failed_input_backing->seed(std::as_bytes(std::span{seeded})) ||
      !replacement_input_backing->seed(
          std::as_bytes(std::span{replacement_seed}))) {
    return 9;
  }
  auto failed_input =
      virtual_buffer<std::int32_t>(PageElements, failed_input_backing);
  auto failed_output =
      virtual_buffer<std::int32_t>(PageElements, failed_output_backing);
  auto replacement_input =
      virtual_buffer<std::int32_t>(PageElements, replacement_input_backing);
  auto replacement_output =
      virtual_buffer<std::int32_t>(PageElements, replacement_output_backing);
  auto failed_pipeline =
      failed_input && failed_output
          ? virtual_pipeline(*program, *failed_input, *failed_output, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  auto replacement_pipeline =
      replacement_input && replacement_output
          ? virtual_pipeline(*program, *replacement_input, *replacement_output,
                             config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  failed_output_backing->fail_next_write(Reason::BackendFailed);
  if (!failed_pipeline || !replacement_pipeline ||
      failed_pipeline->run().reason() != Reason::BackendFailed ||
      !replacement_pipeline->run() ||
      replacement_output_backing->facts().write_count != 1u) {
    return 10;
  }
  std::array<std::int32_t, PageElements> replacement_observed{};
  if (!replacement_output_backing->observe(
          std::as_writable_bytes(std::span{replacement_observed}))) {
    return 11;
  }
  for (std::size_t index = 0u; index < replacement_observed.size(); ++index) {
    if (replacement_observed[index] != (replacement_seed[index] + 5) * 3) {
      return 12;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
