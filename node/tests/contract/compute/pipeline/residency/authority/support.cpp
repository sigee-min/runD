#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

namespace rund_node_test_pipeline_residency::authority_detail {

const rund::compute::detail::residency::CacheUse &RegionA() {
  static const rund::compute::detail::residency::CacheUse region_a{
      .key = {.backing = 20u, .version = 1u, .page = 0u},
      .access = rund::compute::detail::residency::Access::Read,
      .next_use = rund::compute::detail::residency::NeverUse,
  };
  return region_a;
}

} // namespace rund_node_test_pipeline_residency::authority_detail
