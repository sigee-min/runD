#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <new>
#include <utility>

namespace rund::compute::resource::plan_detail {
namespace {

struct ConflictContext final {
  const std::vector<PhysicalAccess> *prior{};
  const PhysicalAccess *current{};
  Plan *plan{};
  std::unordered_map<std::uint64_t, std::size_t> *dependency_rows{};
  detail::AnalysisStats *stats{};
};

[[nodiscard]] bool record_conflict(void *const opaque,
                                   const std::size_t before_index) {
  auto &context = *static_cast<ConflictContext *>(opaque);
  const PhysicalAccess &before = (*context.prior)[before_index];
  const PhysicalAccess &current = *context.current;
  if (context.stats != nullptr) {
    ++context.stats->exact_checks;
  }
  const Overlap overlap = find_overlap(before, current);
  if (before.access.node == current.access.node || !overlap.found ||
      (before.access.mode == AccessMode::Read &&
       current.access.mode == AccessMode::Read)) {
    return true;
  }
  const std::uint64_t dependency_key =
      (static_cast<std::uint64_t>(before.access.node) << 32u) |
      current.access.node;
  const Barrier witness{
      .alias_group = current.alias_group,
      .before_resource = before.access.resource,
      .after_resource = current.access.resource,
      .offset_bytes = overlap.begin,
      .size_bytes = overlap.end - overlap.begin,
      .before_offset_bytes = before.offset,
      .before_element_bytes = before.element_bytes,
      .before_element_count = before.element_count,
      .before_stride_bytes = before.stride_bytes,
      .after_offset_bytes = current.offset,
      .after_element_bytes = current.element_bytes,
      .after_element_count = current.element_count,
      .after_stride_bytes = current.stride_bytes,
      .before_node = before.access.node,
      .after_node = current.access.node,
      .before = before.access.mode,
      .after = current.access.mode,
  };
  const auto [row, inserted] = context.dependency_rows->try_emplace(
      dependency_key, context.plan->dependencies.size());
  if (inserted) {
    context.plan->dependencies.push_back(
        Dependency{before.access.node, current.access.node});
    context.plan->barriers.push_back(witness);
  } else if (context.plan->barriers[row->second].before_resource ==
                 context.plan->barriers[row->second].after_resource &&
             witness.before_resource != witness.after_resource) {
    context.plan->barriers[row->second] = witness;
  }
  // A complete later overlap is the visibility frontier; older edges are
  // transitive command-order mirrors, not new synchronization.
  return overlap.begin != current.offset || overlap.end != current.envelope_end;
}

} // namespace

Result<Plan> analyze_indexed(const std::span<const Resource> resources,
                             const std::span<const Access> accesses,
                             const std::uint32_t node_count,
                             detail::AnalysisStats *const stats) {
  if (resources.empty() || (node_count == 0u && !accesses.empty())) {
    return Result<Plan>::fail(Reason::ResourceGraphIncomplete);
  }
  try {
    Plan plan;
    plan.lifetimes.reserve(resources.size());
    plan.dependencies.reserve(accesses.size());
    plan.barriers.reserve(accesses.size());
    std::unordered_map<std::uint64_t, std::size_t> alias_groups;
    alias_groups.reserve(resources.size());
    std::vector<std::size_t> resource_groups(resources.size());
    std::vector<std::size_t> group_roots;
    group_roots.reserve(resources.size());
    for (std::size_t index = 0u; index < resources.size(); ++index) {
      const Resource &resource = resources[index];
      if (resource.id != index + 1u || resource.alias_group == 0u ||
          !rund::kernel::checked::add(resource.alias_offset_bytes,
                                      resource.bytes)) {
        return Result<Plan>::fail(Reason::ResourceInvalid);
      }
      plan.lifetimes.emplace_back();
      const auto [group, inserted] =
          alias_groups.try_emplace(resource.alias_group, group_roots.size());
      if (inserted) {
        group_roots.push_back(NoAccess);
      }
      resource_groups[index] = group->second;
    }

    std::vector<PhysicalAccess> prior;
    prior.reserve(accesses.size());
    std::unordered_map<std::uint64_t, std::size_t> dependency_rows;
    std::vector<Candidate> candidates;
    candidates.reserve(64u);
    dependency_rows.reserve(accesses.size());
    std::uint32_t previous_node = 0u;
    bool first = true;
    for (const Access &access : accesses) {
      const Resource *const resource =
          find_resource(resources, access.resource);
      if (resource == nullptr || access.node >= node_count ||
          (!first && access.node < previous_node)) {
        return Result<Plan>::fail(Reason::ResourceAccessInvalid);
      }
      first = false;
      previous_node = access.node;
      auto materialized = physical_access(*resource, access);
      if (!materialized) {
        return Result<Plan>::fail(materialized.reason());
      }
      PhysicalAccess current = std::move(materialized).value();
      Lifetime &lifetime = plan.lifetimes[access.resource - 1u];
      lifetime.first_use = std::min(lifetime.first_use, access.node);
      lifetime.last_use = access.node;

      const std::size_t group = resource_groups[access.resource - 1u];
      ConflictContext context{
          .prior = &prior,
          .current = &current,
          .plan = &plan,
          .dependency_rows = &dependency_rows,
          .stats = stats,
      };
      visit_candidates(prior, group_roots[group], current.offset,
                       current.envelope_end, candidates, stats, record_conflict,
                       &context);
      const std::size_t current_index = prior.size();
      current.subtree_latest = current_index;
      prior.push_back(current);
      group_roots[group] =
          insert_access(prior, group_roots[group], current_index, stats);
    }
    return Result<Plan>::success(std::move(plan));
  } catch (const std::bad_alloc &) {
    return Result<Plan>::fail(Reason::ResourceGraphCapacity);
  }
}

} // namespace rund::compute::resource::plan_detail

namespace rund::compute::resource {

Result<Plan> analyze(const std::span<const Resource> resources,
                     const std::span<const Access> accesses,
                     const std::uint32_t node_count) {
  return plan_detail::analyze_indexed(resources, accesses, node_count, nullptr);
}

Result<Plan> detail::analyze_measured(const std::span<const Resource> resources,
                                      const std::span<const Access> accesses,
                                      const std::uint32_t node_count,
                                      AnalysisStats &stats) {
  stats = {};
  return plan_detail::analyze_indexed(resources, accesses, node_count, &stats);
}

} // namespace rund::compute::resource
