#include "local.hpp"

#include "../../../../../src/compute/resource/index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_graph_services {
namespace {

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

[[nodiscard]] std::uint32_t reverse_bits(std::uint32_t value,
                                         const std::uint32_t width) noexcept {
  std::uint32_t reversed = 0u;
  for (std::uint32_t bit = 0u; bit < width; ++bit) {
    reversed = (reversed << 1u) | (value & 1u);
    value >>= 1u;
  }
  return reversed;
}

[[nodiscard]] bool bounded_disjoint_work() {
  using rund::compute::resource::detail::AnalysisStats;
  using rund::compute::resource::detail::analyze_measured;
  constexpr std::uint32_t Width = 12u;
  constexpr std::uint32_t Count = 1u << Width;
  const std::array<Resource, 1u> resources{Resource{
      .id = 1u,
      .bytes = static_cast<std::uint64_t>(Count) * 2u,
      .alias_group = 47u,
      .alias_offset_bytes = 0u,
  }};
  std::vector<Access> accesses;
  accesses.reserve(Count);
  for (std::uint32_t index = 0u; index < Count; ++index) {
    accesses.push_back(Access{
        .node = index,
        .resource = 1u,
        .mode = AccessMode::Write,
        .offset_bytes =
            static_cast<std::uint64_t>(reverse_bits(index, Width)) * 2u,
        .size_bytes = 1u,
    });
  }
  AnalysisStats stats;
  const auto plan = analyze_measured(resources, accesses, Count, stats);
  constexpr std::uint64_t LogBound = Width + 1u;
  constexpr std::uint64_t IndexedBound =
      4u * static_cast<std::uint64_t>(Count) * LogBound;
  constexpr std::uint64_t BrutePairs =
      static_cast<std::uint64_t>(Count) * (Count - 1u) / 2u;
  const std::uint64_t indexed_visits = stats.insert_visits + stats.query_visits;
  return plan && plan->dependencies.empty() && plan->barriers.empty() &&
         stats.envelope_candidates == 0u && stats.exact_checks == 0u &&
         indexed_visits <= IndexedBound && indexed_visits * 32u < BrutePairs;
}

[[nodiscard]] bool bounded_frontier_work() {
  using rund::compute::resource::detail::AnalysisStats;
  using rund::compute::resource::detail::analyze_measured;
  constexpr std::uint32_t Width = 12u;
  constexpr std::uint32_t Count = 1u << Width;
  const std::array<Resource, 1u> resources{Resource{
      .id = 1u,
      .bytes = 64u,
      .alias_group = 53u,
      .alias_offset_bytes = 0u,
  }};
  std::vector<Access> accesses;
  accesses.reserve(Count);
  for (std::uint32_t index = 0u; index < Count; ++index) {
    accesses.push_back(Access{
        .node = index,
        .resource = 1u,
        .mode = AccessMode::Write,
        .offset_bytes = 0u,
        .size_bytes = 64u,
    });
  }
  AnalysisStats stats;
  const auto plan = analyze_measured(resources, accesses, Count, stats);
  constexpr std::uint64_t LogBound = Width + 1u;
  constexpr std::uint64_t IndexedBound =
      8u * static_cast<std::uint64_t>(Count) * LogBound;
  const std::uint64_t indexed_visits = stats.insert_visits + stats.query_visits;
  return plan && plan->dependencies.size() == Count - 1u &&
         plan->barriers.size() == Count - 1u &&
         stats.envelope_candidates == Count - 1u &&
         stats.exact_checks == Count - 1u && indexed_visits <= IndexedBound;
}

} // namespace

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
