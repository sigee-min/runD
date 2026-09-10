#pragma once

#include "../backing.hpp"
#include "../model.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund_node_test_virtual::product::cache {

template <typename Program, typename SecondPipeline>
[[nodiscard]] inline int
CheckDiscard(Program &program, const rund::compute::ResidencyConfig &config,
             const std::array<std::int32_t, PageElements> &seeded,
             SecondPipeline &second_pipeline) {
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
  auto failed_input = rund::compute::virtual_buffer<std::int32_t>(
      PageElements, failed_input_backing);
  auto failed_output = rund::compute::virtual_buffer<std::int32_t>(
      PageElements, failed_output_backing);
  auto replacement_input = rund::compute::virtual_buffer<std::int32_t>(
      PageElements, replacement_input_backing);
  auto replacement_output = rund::compute::virtual_buffer<std::int32_t>(
      PageElements, replacement_output_backing);
  auto failed_pipeline =
      failed_input && failed_output
          ? rund::compute::virtual_pipeline(*program, *failed_input,
                                            *failed_output, config)
          : rund::compute::Result<rund::compute::VirtualPipeline<std::int32_t(
                std::int32_t)>>::fail(rund::compute::Reason::PipelineInvalid);
  auto replacement_pipeline =
      replacement_input && replacement_output
          ? rund::compute::virtual_pipeline(*program, *replacement_input,
                                            *replacement_output, config)
          : rund::compute::Result<rund::compute::VirtualPipeline<std::int32_t(
                std::int32_t)>>::fail(rund::compute::Reason::PipelineInvalid);
  failed_output_backing->fail_next_write(rund::compute::Reason::BackendFailed);
  if (!failed_pipeline || !replacement_pipeline ||
      failed_pipeline->run().reason() != rund::compute::Reason::BackendFailed ||
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
  if (!second_pipeline->run() ||
      second_pipeline->stats().pipeline.residency.page_in_count != 1u) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::cache
