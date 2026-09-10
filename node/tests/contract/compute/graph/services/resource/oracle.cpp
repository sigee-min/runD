#include "local.hpp"

#include "../../../../../../src/compute/resource/index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_graph_services::resource_test_detail {

using rund::compute::resource::Access;
using rund::compute::resource::AccessMode;
using rund::compute::resource::Barrier;
using rund::compute::resource::Dependency;
using rund::compute::resource::Lifetime;
using rund::compute::resource::NoNode;
using rund::compute::resource::Plan;
using rund::compute::resource::Resource;

struct FlatAccess final {
  Access access{};
  std::uint64_t alias_group{};
  std::uint64_t offset{};
  std::uint64_t end{};
};

[[nodiscard]] bool same_plan(const Plan &left, const Plan &right) {
  if (left.lifetimes.size() != right.lifetimes.size() ||
      left.dependencies != right.dependencies ||
      left.barriers != right.barriers) {
    return false;
  }
  for (std::size_t index = 0u; index < left.lifetimes.size(); ++index) {
    if (left.lifetimes[index].first_use != right.lifetimes[index].first_use ||
        left.lifetimes[index].last_use != right.lifetimes[index].last_use) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Plan brute_plan(const std::span<const Resource> resources,
                              const std::span<const Access> accesses) {
  Plan plan;
  plan.lifetimes.resize(resources.size());
  std::vector<FlatAccess> prior;
  prior.reserve(accesses.size());
  for (const Access &access : accesses) {
    const Resource &resource = resources[access.resource - 1u];
    const FlatAccess current{
        .access = access,
        .alias_group = resource.alias_group,
        .offset = resource.alias_offset_bytes + access.offset_bytes,
        .end = resource.alias_offset_bytes + access.offset_bytes +
               access.size_bytes,
    };
    Lifetime &lifetime = plan.lifetimes[access.resource - 1u];
    lifetime.first_use = std::min(lifetime.first_use, access.node);
    lifetime.last_use = access.node;
    for (std::size_t index = prior.size(); index != 0u; --index) {
      const FlatAccess &before = prior[index - 1u];
      if (before.alias_group != current.alias_group ||
          before.offset >= current.end || current.offset >= before.end ||
          before.access.node == current.access.node ||
          (before.access.mode == AccessMode::Read &&
           current.access.mode == AccessMode::Read)) {
        continue;
      }
      const std::uint64_t overlap_begin =
          std::max(before.offset, current.offset);
      const std::uint64_t overlap_end = std::min(before.end, current.end);
      const Dependency dependency{.before_node = before.access.node,
                                  .after_node = current.access.node};
      const Barrier barrier{
          .alias_group = current.alias_group,
          .before_resource = before.access.resource,
          .after_resource = current.access.resource,
          .offset_bytes = overlap_begin,
          .size_bytes = overlap_end - overlap_begin,
          .before_offset_bytes = before.offset,
          .before_element_bytes = 1u,
          .before_element_count = before.access.size_bytes,
          .before_stride_bytes = 1u,
          .after_offset_bytes = current.offset,
          .after_element_bytes = 1u,
          .after_element_count = current.access.size_bytes,
          .after_stride_bytes = 1u,
          .before_node = before.access.node,
          .after_node = current.access.node,
          .before = before.access.mode,
          .after = current.access.mode,
      };
      const auto row = std::find(plan.dependencies.begin(),
                                 plan.dependencies.end(), dependency);
      if (row == plan.dependencies.end()) {
        plan.dependencies.push_back(dependency);
        plan.barriers.push_back(barrier);
      } else {
        Barrier &retained = plan.barriers[static_cast<std::size_t>(
            row - plan.dependencies.begin())];
        if (retained.before_resource == retained.after_resource &&
            barrier.before_resource != barrier.after_resource) {
          retained = barrier;
        }
      }
      if (overlap_begin == current.offset && overlap_end == current.end) {
        break;
      }
    }
    prior.push_back(current);
  }
  return plan;
}

struct Random final {
  std::uint64_t state{0x6a09e667f3bcc909ull};

  [[nodiscard]] std::uint64_t next() noexcept {
    state ^= state << 13u;
    state ^= state >> 7u;
    state ^= state << 17u;
    return state;
  }

  [[nodiscard]] std::uint64_t below(const std::uint64_t bound) noexcept {
    return next() % bound;
  }
};

[[nodiscard]] bool matches_brute_force() {
  using rund::compute::resource::analyze;
  const std::array<Resource, 5u> resources{
      Resource{.id = 1u,
               .bytes = 256u,
               .alias_group = 41u,
               .alias_offset_bytes = 0u},
      Resource{.id = 2u,
               .bytes = 256u,
               .alias_group = 41u,
               .alias_offset_bytes = 64u},
      Resource{.id = 3u,
               .bytes = 256u,
               .alias_group = 41u,
               .alias_offset_bytes = 128u},
      Resource{.id = 4u,
               .bytes = 256u,
               .alias_group = 43u,
               .alias_offset_bytes = 0u},
      Resource{.id = 5u,
               .bytes = 256u,
               .alias_group = 43u,
               .alias_offset_bytes = 96u},
  };
  Random random;
  for (std::uint32_t trial = 0u; trial < 384u; ++trial) {
    const std::size_t count = 1u + random.below(31u);
    std::vector<Access> accesses;
    accesses.reserve(count);
    std::uint32_t node = 0u;
    for (std::size_t index = 0u; index < count; ++index) {
      if (index != 0u && random.below(3u) != 0u) {
        ++node;
      }
      const std::uint32_t resource =
          1u + static_cast<std::uint32_t>(random.below(resources.size()));
      const std::uint64_t bytes = 1u + random.below(32u);
      accesses.push_back(Access{
          .node = node,
          .resource = resource,
          .mode = random.below(2u) == 0u ? AccessMode::Read : AccessMode::Write,
          .offset_bytes = random.below(257u - bytes),
          .size_bytes = bytes,
      });
    }
    const auto indexed = analyze(resources, accesses, node + 1u);
    if (!indexed || !same_plan(*indexed, brute_plan(resources, accesses))) {
      return false;
    }
  }
  return true;
}

} // namespace rund_node_graph_services::resource_test_detail
