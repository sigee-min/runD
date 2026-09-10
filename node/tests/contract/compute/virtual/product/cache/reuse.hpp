#pragma once

#include "../backing.hpp"
#include "../golden.hpp"
#include "../model.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund_node_test_virtual::product::cache {

template <typename FirstPipeline, typename SecondPipeline>
[[nodiscard]] inline int CheckReuse(FirstPipeline &first_pipeline,
                                    SecondPipeline &second_pipeline,
                                    MemoryVirtualBacking &input_backing,
                                    MemoryVirtualBacking &second_backing) {
  if (!first_pipeline || !second_pipeline || !first_pipeline->run()) {
    return 4;
  }
  const BackingFacts after_first = input_backing.facts();
  if (after_first.read_count != 1u || !second_pipeline->run()) {
    return 5;
  }
  const auto pipeline_stats = second_pipeline->stats();
  const auto stats = pipeline_stats.pipeline.residency;
  std::array<std::int32_t, PageElements> observed{};
  if (!second_backing.observe(std::as_writable_bytes(std::span{observed}))) {
    return 6;
  }
  if (!GoldenPageMatches(observed, 0u)) {
    return 7;
  }
  if (input_backing.facts().read_count != after_first.read_count ||
      stats.page_in_count != 0u || stats.cache_hit_count != 1u ||
      stats.late_page_count != 0u || stats.prefetch_count != 0u ||
      stats.page_out_count != 1u) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::cache
