#pragma once

#include "../backing.hpp"
#include "../model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund_node_test_virtual::product::cache {

template <typename Opened, typename SecondPipeline>
[[nodiscard]] inline int CheckLayout(Opened &opened,
                                     SecondPipeline &second_pipeline) {
  using namespace rund::compute;
  auto wide_program = on(*opened)
                          .template map<std::uint64_t>(
                              "virtual-product-global-layout", PageElements,
                              [](auto value) { return value + 1u; })
                          .compile();
  constexpr std::size_t WideBytes = PageElements * sizeof(std::uint64_t);
  auto wide_input_backing =
      std::make_shared<MemoryVirtualBacking>(WideBytes, WideBytes);
  auto wide_output_backing =
      std::make_shared<MemoryVirtualBacking>(WideBytes, WideBytes);
  std::array<std::uint64_t, PageElements> wide_seed{};
  for (std::size_t index = 0u; index < wide_seed.size(); ++index) {
    wide_seed[index] = 1000u + index;
  }
  if (!wide_program ||
      !wide_input_backing->seed(std::as_bytes(std::span{wide_seed}))) {
    return 14;
  }
  auto wide_input =
      virtual_buffer<std::uint64_t>(PageElements, wide_input_backing);
  auto wide_output =
      virtual_buffer<std::uint64_t>(PageElements, wide_output_backing);
  const ResidencyConfig wide_config{
      .device_resident_bytes = WideBytes * 4u,
      .host_resident_bytes = WideBytes * 4u,
  };
  auto wide = wide_input && wide_output
                  ? virtual_pipeline(*wide_program, *wide_input, *wide_output,
                                     wide_config)
                  : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                        Reason::PipelineInvalid);
  std::array<std::uint64_t, PageElements> wide_observed{};
  if (!wide || !wide->run() ||
      !wide_output_backing->observe(
          std::as_writable_bytes(std::span{wide_observed}))) {
    return 15;
  }
  for (std::size_t index = 0u; index < wide_observed.size(); ++index) {
    if (wide_observed[index] != wide_seed[index] + 1u) {
      return 16;
    }
  }
  if (wide->stats().pipeline.residency.page_in_count != 1u ||
      wide->stats().pipeline.residency.cache_hit_count != 0u ||
      !second_pipeline->run() ||
      second_pipeline->stats().pipeline.residency.page_in_count != 0u ||
      second_pipeline->stats().pipeline.residency.cache_hit_count != 1u) {
    return 17;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::cache
