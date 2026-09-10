#include "local.hpp"
#include "resource/local.hpp"

#include "../../../../../src/compute/resource/index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_graph_services {

using resource_test_detail::bounded_disjoint_work;
using resource_test_detail::bounded_frontier_work;
using resource_test_detail::matches_brute_force;

bool ValidResourcePlan() {
  using namespace rund::compute::resource;
  const std::array<Resource, 4u> resources{
      Resource{
          .id = 1u, .bytes = 64u, .alias_group = 7u, .alias_offset_bytes = 0u},
      Resource{
          .id = 2u, .bytes = 64u, .alias_group = 7u, .alias_offset_bytes = 32u},
      Resource{
          .id = 3u, .bytes = 32u, .alias_group = 7u, .alias_offset_bytes = 96u},
      Resource{
          .id = 4u, .bytes = 0u, .alias_group = 8u, .alias_offset_bytes = 0u},
  };
  const std::array<Access, 5u> accesses{
      Access{.node = 0u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .element_bytes = 1u,
             .element_count = 32u,
             .stride_bytes = 1u},
      Access{.node = 1u,
             .resource = 2u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .element_bytes = 1u,
             .element_count = 16u,
             .stride_bytes = 1u},
      Access{.node = 2u,
             .resource = 2u,
             .mode = AccessMode::Write,
             .offset_bytes = 32u,
             .element_bytes = 1u,
             .element_count = 16u,
             .stride_bytes = 1u},
      Access{.node = 3u,
             .resource = 1u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .element_bytes = 1u,
             .element_count = 8u,
             .stride_bytes = 1u},
      Access{.node = 4u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .element_bytes = 1u,
             .element_count = 8u,
             .stride_bytes = 1u},
  };
  auto plan = analyze(resources, accesses, 5u);
  if (!plan || plan->lifetimes.size() != 4u ||
      plan->lifetimes[0u].first_use != 0u ||
      plan->lifetimes[0u].last_use != 4u ||
      plan->lifetimes[1u].first_use != 1u ||
      plan->lifetimes[1u].last_use != 2u ||
      plan->lifetimes[2u].first_use != NoNode ||
      plan->lifetimes[3u].first_use != NoNode ||
      plan->dependencies !=
          std::vector<Dependency>{{.before_node = 0u, .after_node = 1u},
                                  {.before_node = 0u, .after_node = 4u}} ||
      plan->barriers.size() != 2u) {
    return false;
  }
  const std::array<Access, 5u> contiguous_accesses{
      Access{.node = 0u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .size_bytes = 32u},
      Access{.node = 1u,
             .resource = 2u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .size_bytes = 16u},
      Access{.node = 2u,
             .resource = 2u,
             .mode = AccessMode::Write,
             .offset_bytes = 32u,
             .size_bytes = 16u},
      Access{.node = 3u,
             .resource = 1u,
             .mode = AccessMode::Read,
             .offset_bytes = 0u,
             .size_bytes = 8u},
      Access{.node = 4u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 16u,
             .size_bytes = 8u},
  };
  const auto contiguous_plan = analyze(resources, contiguous_accesses, 5u);
  if (!contiguous_plan || contiguous_plan->dependencies != plan->dependencies ||
      contiguous_plan->barriers != plan->barriers) {
    return false;
  }
  const Barrier &alias = plan->barriers.front();
  if (alias.alias_group != 7u || alias.before_resource != 1u ||
      alias.after_resource != 2u || alias.offset_bytes != 32u ||
      alias.size_bytes != 16u || alias.before_node != 0u ||
      alias.after_node != 1u || alias.before != AccessMode::Write ||
      alias.after != AccessMode::Read || alias.before_offset_bytes != 16u ||
      alias.before_element_bytes != 1u || alias.before_element_count != 32u ||
      alias.before_stride_bytes != 1u || alias.after_offset_bytes != 32u ||
      alias.after_element_bytes != 1u || alias.after_element_count != 16u ||
      alias.after_stride_bytes != 1u) {
    return false;
  }
  const std::span<const Access> zero_accesses;
  const auto zero = analyze(resources, zero_accesses, 0u);
  if (!zero || zero->lifetimes.size() != resources.size() ||
      !zero->dependencies.empty() || !zero->barriers.empty() ||
      std::any_of(zero->lifetimes.begin(), zero->lifetimes.end(),
                  [](const Lifetime &lifetime) {
                    return lifetime.first_use != NoNode ||
                           lifetime.last_use != NoNode;
                  })) {
    return false;
  }
  auto invalid = accesses;
  invalid[0u].offset_bytes = 48u;
  invalid[0u].element_count = 32u;
  const auto rejected = analyze(resources, invalid, 5u);
  invalid = accesses;
  invalid[0u].mode = static_cast<AccessMode>(255u);
  const auto invalid_mode = analyze(resources, invalid, 5u);
  if (rejected || rejected.error() != "compute_resource_access_invalid" ||
      invalid_mode ||
      invalid_mode.error() != "compute_resource_access_invalid") {
    return false;
  }

  // More than 64 elements selects the count-independent Diophantine path.
  // Even and odd lanes share an envelope but have no byte intersection.
  const std::array<Resource, 1u> strided_resources{Resource{
      .id = 1u, .bytes = 640u, .alias_group = 9u, .alias_offset_bytes = 0u}};
  const std::array<Access, 2u> disjoint_strides{
      Access{.node = 0u,
             .resource = 1u,
             .mode = AccessMode::Write,
             .offset_bytes = 0u,
             .element_bytes = 4u,
             .element_count = 80u,
             .stride_bytes = 8u},
      Access{.node = 1u,
             .resource = 1u,
             .mode = AccessMode::Read,
             .offset_bytes = 4u,
             .element_bytes = 4u,
             .element_count = 80u,
             .stride_bytes = 8u},
  };
  const auto disjoint = analyze(strided_resources, disjoint_strides, 2u);
  const auto disjoint_pair =
      intersects(strided_resources[0u], disjoint_strides[0u],
                 strided_resources[0u], disjoint_strides[1u]);
  if (!disjoint || !disjoint->dependencies.empty() ||
      !disjoint->barriers.empty() || !disjoint_pair || *disjoint_pair) {
    return false;
  }
  const std::array<Access, 2u> overlapping_strides{
      disjoint_strides[0u],
      Access{.node = 1u,
             .resource = 1u,
             .mode = AccessMode::Read,
             .offset_bytes = 8u,
             .element_bytes = 4u,
             .element_count = 79u,
             .stride_bytes = 8u},
  };
  const auto overlapping = analyze(strided_resources, overlapping_strides, 2u);
  const auto overlapping_pair =
      intersects(strided_resources[0u], overlapping_strides[0u],
                 strided_resources[0u], overlapping_strides[1u]);
  if (!overlapping ||
      overlapping->dependencies !=
          std::vector<Dependency>{{.before_node = 0u, .after_node = 1u}} ||
      overlapping->barriers.size() != 1u || !overlapping_pair ||
      !*overlapping_pair) {
    return false;
  }
  const Barrier &strided = overlapping->barriers.front();
  if (strided.offset_bytes != 8u || strided.size_bytes != 4u ||
      strided.before_offset_bytes != 0u || strided.before_element_bytes != 4u ||
      strided.before_element_count != 80u ||
      strided.before_stride_bytes != 8u || strided.after_offset_bytes != 8u ||
      strided.after_element_bytes != 4u || strided.after_element_count != 79u ||
      strided.after_stride_bytes != 8u) {
    return false;
  }
  auto invalid_stride = disjoint_strides;
  invalid_stride[0u].stride_bytes = 2u;
  auto invalid_envelope = disjoint_strides;
  invalid_envelope[1u].offset_bytes = 8u;
  const auto invalid_pair =
      intersects(strided_resources[0u], invalid_stride[0u],
                 strided_resources[0u], invalid_stride[1u]);
  return !analyze(strided_resources, invalid_stride, 2u) && !invalid_pair &&
         invalid_pair.reason() ==
             rund::compute::Reason::ResourceAccessInvalid &&
         !analyze(strided_resources, invalid_envelope, 2u) &&
         matches_brute_force() && bounded_disjoint_work() &&
         bounded_frontier_work();
}

} // namespace rund_node_graph_services
