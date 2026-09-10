#include "local.hpp"

#include <rund/compute.hpp>

#include <array>
#include <vector>

namespace package_compute::graph_services {

int CheckResourcePlan() {
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
  const auto plan = analyze(resources, accesses, 5u);
  if (!plan) {
    return plan.exit_code();
  }
  if (plan->lifetimes.size() != resources.size() ||
      plan->lifetimes[0u].first_use != 0u ||
      plan->lifetimes[0u].last_use != 4u ||
      plan->lifetimes[1u].first_use != 1u ||
      plan->lifetimes[1u].last_use != 2u ||
      plan->lifetimes[2u].first_use != NoNode ||
      plan->lifetimes[2u].last_use != NoNode ||
      plan->lifetimes[3u].first_use != NoNode ||
      plan->lifetimes[3u].last_use != NoNode ||
      plan->dependencies !=
          std::vector<Dependency>{{.before_node = 0u, .after_node = 1u},
                                  {.before_node = 0u, .after_node = 4u}} ||
      plan->barriers.size() != 2u) {
    return 2;
  }
  const Barrier &alias = plan->barriers.front();
  if (alias.alias_group != 7u || alias.before_resource != 1u ||
      alias.after_resource != 2u || alias.offset_bytes != 32u ||
      alias.size_bytes != 16u || alias.before_node != 0u ||
      alias.after_node != 1u || alias.before != AccessMode::Write ||
      alias.after != AccessMode::Read) {
    return 2;
  }
  return 0;
}

} // namespace package_compute::graph_services
