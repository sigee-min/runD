#include <rund/compute/resource/plan.hpp>

#include <kernel/core/checked.hpp>

#include "index.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>

namespace rund::compute::resource {
namespace {

[[nodiscard]] const Resource *
find_resource(const std::span<const Resource> resources,
              const std::uint32_t id) noexcept {
  return id == 0u || id > resources.size() ? nullptr : &resources[id - 1u];
}

struct PhysicalAccess final {
  Access access{};
  std::uint64_t alias_group{};
  std::uint64_t offset{};
  std::uint64_t element_bytes{};
  std::uint64_t element_count{};
  std::uint64_t stride_bytes{};
  std::uint64_t envelope_end{};
  std::uint64_t subtree_begin{};
  std::uint64_t subtree_end{};
  std::size_t subtree_latest{};
  std::size_t left{};
  std::size_t right{};
  std::uint32_t height{1u};
};

inline constexpr std::size_t NoAccess = std::numeric_limits<std::size_t>::max();

struct Overlap final {
  std::uint64_t begin{};
  std::uint64_t end{};
  bool found{};
};

[[nodiscard]] std::uint32_t height(const std::vector<PhysicalAccess> &accesses,
                                   const std::size_t index) noexcept {
  return index == NoAccess ? 0u : accesses[index].height;
}

void refresh(std::vector<PhysicalAccess> &accesses,
             const std::size_t index) noexcept {
  PhysicalAccess &access = accesses[index];
  access.height = 1u + std::max(height(accesses, access.left),
                                height(accesses, access.right));
  access.subtree_begin = access.offset;
  access.subtree_end = access.envelope_end;
  access.subtree_latest = index;
  if (access.left != NoAccess) {
    access.subtree_begin =
        std::min(access.subtree_begin, accesses[access.left].subtree_begin);
    access.subtree_end =
        std::max(access.subtree_end, accesses[access.left].subtree_end);
    access.subtree_latest =
        std::max(access.subtree_latest, accesses[access.left].subtree_latest);
  }
  if (access.right != NoAccess) {
    access.subtree_begin =
        std::min(access.subtree_begin, accesses[access.right].subtree_begin);
    access.subtree_end =
        std::max(access.subtree_end, accesses[access.right].subtree_end);
    access.subtree_latest =
        std::max(access.subtree_latest, accesses[access.right].subtree_latest);
  }
}

[[nodiscard]] bool before_key(const PhysicalAccess &left,
                              const std::size_t left_index,
                              const PhysicalAccess &right,
                              const std::size_t right_index) noexcept {
  return left.offset < right.offset ||
         (left.offset == right.offset && left_index < right_index);
}

[[nodiscard]] std::size_t rotate_left(std::vector<PhysicalAccess> &accesses,
                                      const std::size_t root) noexcept {
  const std::size_t pivot = accesses[root].right;
  const std::size_t middle = accesses[pivot].left;
  accesses[pivot].left = root;
  accesses[root].right = middle;
  refresh(accesses, root);
  refresh(accesses, pivot);
  return pivot;
}

[[nodiscard]] std::size_t rotate_right(std::vector<PhysicalAccess> &accesses,
                                       const std::size_t root) noexcept {
  const std::size_t pivot = accesses[root].left;
  const std::size_t middle = accesses[pivot].right;
  accesses[pivot].right = root;
  accesses[root].left = middle;
  refresh(accesses, root);
  refresh(accesses, pivot);
  return pivot;
}

[[nodiscard]] std::size_t
insert_access(std::vector<PhysicalAccess> &accesses, const std::size_t root,
              const std::size_t inserted,
              detail::AnalysisStats *const stats) noexcept {
  if (root == NoAccess) {
    return inserted;
  }
  if (stats != nullptr) {
    ++stats->insert_visits;
  }
  if (before_key(accesses[inserted], inserted, accesses[root], root)) {
    accesses[root].left =
        insert_access(accesses, accesses[root].left, inserted, stats);
  } else {
    accesses[root].right =
        insert_access(accesses, accesses[root].right, inserted, stats);
  }
  refresh(accesses, root);
  const int balance = static_cast<int>(height(accesses, accesses[root].left)) -
                      static_cast<int>(height(accesses, accesses[root].right));
  if (balance > 1) {
    if (!before_key(accesses[inserted], inserted, accesses[accesses[root].left],
                    accesses[root].left)) {
      accesses[root].left = rotate_left(accesses, accesses[root].left);
    }
    return rotate_right(accesses, root);
  }
  if (balance < -1) {
    if (before_key(accesses[inserted], inserted, accesses[accesses[root].right],
                   accesses[root].right)) {
      accesses[root].right = rotate_right(accesses, accesses[root].right);
    }
    return rotate_left(accesses, root);
  }
  return root;
}

struct Candidate final {
  std::size_t root{};
  std::size_t latest{};
  bool single{};
};

[[nodiscard]] bool older(const Candidate left, const Candidate right) noexcept {
  return left.latest < right.latest;
}

template <class Visit>
void visit_candidates(const std::vector<PhysicalAccess> &accesses,
                      const std::size_t root, const std::uint64_t begin,
                      const std::uint64_t end, std::vector<Candidate> &queue,
                      detail::AnalysisStats *const stats, Visit &&visit) {
  const auto push_subtree = [&](const std::size_t index) {
    if (index == NoAccess) {
      return;
    }
    const PhysicalAccess &access = accesses[index];
    if (access.subtree_end <= begin || access.subtree_begin >= end) {
      return;
    }
    queue.push_back(Candidate{
        .root = index,
        .latest = access.subtree_latest,
    });
    std::push_heap(queue.begin(), queue.end(), older);
  };
  const auto push_single = [&](const std::size_t index) {
    const PhysicalAccess &access = accesses[index];
    if (access.envelope_end <= begin || access.offset >= end) {
      return;
    }
    queue.push_back(Candidate{
        .root = index,
        .latest = index,
        .single = true,
    });
    std::push_heap(queue.begin(), queue.end(), older);
  };

  queue.clear();
  push_subtree(root);
  while (!queue.empty()) {
    std::pop_heap(queue.begin(), queue.end(), older);
    const Candidate current = queue.back();
    queue.pop_back();
    if (stats != nullptr) {
      ++stats->query_visits;
    }
    if (current.single) {
      if (stats != nullptr) {
        ++stats->envelope_candidates;
      }
      if (!visit(current.root)) {
        return;
      }
      continue;
    }
    const PhysicalAccess &access = accesses[current.root];
    push_subtree(access.left);
    push_subtree(access.right);
    push_single(current.root);
  }
}

using Wide = __int128_t;

struct Bezout final {
  std::uint64_t gcd{};
  Wide left{};
  Wide right{};
};

[[nodiscard]] Bezout bezout(std::uint64_t left, std::uint64_t right) noexcept {
  Wide old_left = 1;
  Wide next_left = 0;
  Wide old_right = 0;
  Wide next_right = 1;
  while (right != 0u) {
    const std::uint64_t quotient = left / right;
    const std::uint64_t remainder = left % right;
    left = right;
    right = remainder;
    const Wide left_coefficient =
        old_left - static_cast<Wide>(quotient) * next_left;
    old_left = next_left;
    next_left = left_coefficient;
    const Wide right_coefficient =
        old_right - static_cast<Wide>(quotient) * next_right;
    old_right = next_right;
    next_right = right_coefficient;
  }
  return Bezout{.gcd = left, .left = old_left, .right = old_right};
}

[[nodiscard]] Wide floor_div(const Wide value, const Wide divisor) noexcept {
  Wide quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

[[nodiscard]] Wide ceil_div(const Wide value, const Wide divisor) noexcept {
  return -floor_div(-value, divisor);
}

[[nodiscard]] bool bounded_solution(const std::uint64_t left_stride,
                                    const std::uint64_t left_count,
                                    const std::uint64_t right_stride,
                                    const std::uint64_t right_count,
                                    const Wide target,
                                    std::uint64_t &left_index,
                                    std::uint64_t &right_index) noexcept {
  const Bezout coefficients = bezout(left_stride, right_stride);
  if (coefficients.gcd == 0u ||
      target % static_cast<Wide>(coefficients.gcd) != 0) {
    return false;
  }
  const Wide scale = target / static_cast<Wide>(coefficients.gcd);
  const Wide first_left = coefficients.left * scale;
  const Wide first_right = -coefficients.right * scale;
  const Wide left_step = static_cast<Wide>(right_stride / coefficients.gcd);
  const Wide right_step = static_cast<Wide>(left_stride / coefficients.gcd);
  const Wide lower = std::max(ceil_div(-first_left, left_step),
                              ceil_div(-first_right, right_step));
  const Wide upper = std::min(
      floor_div(static_cast<Wide>(left_count - 1u) - first_left, left_step),
      floor_div(static_cast<Wide>(right_count - 1u) - first_right, right_step));
  if (lower > upper) {
    return false;
  }
  const Wide solved_left = first_left + left_step * lower;
  const Wide solved_right = first_right + right_step * lower;
  if (solved_left < 0 || solved_right < 0 ||
      solved_left >= static_cast<Wide>(left_count) ||
      solved_right >= static_cast<Wide>(right_count)) {
    return false;
  }
  left_index = static_cast<std::uint64_t>(solved_left);
  right_index = static_cast<std::uint64_t>(solved_right);
  return true;
}

[[nodiscard]] bool enumerated_solution(const PhysicalAccess &small,
                                       const PhysicalAccess &large,
                                       const Wide delta,
                                       const bool small_is_left,
                                       std::uint64_t &left_index,
                                       std::uint64_t &right_index) noexcept {
  for (std::uint64_t index = 0u; index < small.element_count; ++index) {
    const Wide small_start = static_cast<Wide>(small.offset) +
                             static_cast<Wide>(index) * small.stride_bytes;
    // left_start - right_start == delta.
    const Wide large_start =
        small_is_left ? small_start - delta : small_start + delta;
    if (large_start < static_cast<Wide>(large.offset)) {
      continue;
    }
    const Wide distance = large_start - static_cast<Wide>(large.offset);
    if (distance % static_cast<Wide>(large.stride_bytes) != 0) {
      continue;
    }
    const Wide large_index = distance / large.stride_bytes;
    if (large_index < 0 ||
        large_index >= static_cast<Wide>(large.element_count)) {
      continue;
    }
    if (small_is_left) {
      left_index = index;
      right_index = static_cast<std::uint64_t>(large_index);
    } else {
      left_index = static_cast<std::uint64_t>(large_index);
      right_index = index;
    }
    return true;
  }
  return false;
}

[[nodiscard]] Overlap find_overlap(const PhysicalAccess &left,
                                   const PhysicalAccess &right) noexcept {
  if (left.alias_group != right.alias_group ||
      left.offset >= right.envelope_end || right.offset >= left.envelope_end) {
    return {};
  }
  // Preserve the original contiguous-range contract, including its complete
  // overlap interval. The exact strided form is also contiguous when its
  // stride equals its element width, so both public authoring forms
  // canonicalize to the same dependency and barrier description.
  if (left.stride_bytes == left.element_bytes &&
      right.stride_bytes == right.element_bytes) {
    return Overlap{.begin = std::max(left.offset, right.offset),
                   .end = std::min(left.envelope_end, right.envelope_end),
                   .found = true};
  }
  constexpr std::uint64_t EnumerateLimit = 64u;
  const Wide delta_begin = 1 - static_cast<Wide>(left.element_bytes);
  const Wide delta_end = static_cast<Wide>(right.element_bytes) - 1;
  for (Wide delta = delta_begin; delta <= delta_end; ++delta) {
    std::uint64_t left_index = 0u;
    std::uint64_t right_index = 0u;
    bool found = false;
    if (left.element_count <= EnumerateLimit ||
        right.element_count <= EnumerateLimit) {
      const bool left_is_small = left.element_count <= right.element_count;
      found = left_is_small ? enumerated_solution(left, right, delta, true,
                                                  left_index, right_index)
                            : enumerated_solution(right, left, delta, false,
                                                  left_index, right_index);
    } else {
      const Wide target = static_cast<Wide>(right.offset) - left.offset + delta;
      found = bounded_solution(left.stride_bytes, left.element_count,
                               right.stride_bytes, right.element_count, target,
                               left_index, right_index);
    }
    if (!found) {
      continue;
    }
    const std::uint64_t left_start =
        left.offset + left_index * left.stride_bytes;
    const std::uint64_t right_start =
        right.offset + right_index * right.stride_bytes;
    return Overlap{
        .begin = std::max(left_start, right_start),
        .end = std::min(left_start + left.element_bytes,
                        right_start + right.element_bytes),
        .found = true,
    };
  }
  return {};
}

[[nodiscard]] Result<PhysicalAccess>
physical_access(const Resource &resource, const Access &access) noexcept {
  if (resource.id == 0u || resource.alias_group == 0u ||
      !rund::kernel::checked::add(resource.alias_offset_bytes,
                                  resource.bytes)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceInvalid);
  }
  const bool contiguous = access.size_bytes != 0u;
  const bool strided = access.element_bytes != 0u ||
                       access.element_count != 0u || access.stride_bytes != 0u;
  if (contiguous == strided) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  const std::uint64_t element_bytes = contiguous ? 1u : access.element_bytes;
  const std::uint64_t element_count =
      contiguous ? access.size_bytes : access.element_count;
  const std::uint64_t stride_bytes = contiguous ? 1u : access.stride_bytes;
  std::uint64_t distance = 0u;
  std::uint64_t last = 0u;
  std::uint64_t relative_end = 0u;
  const bool footprint_overflow =
      element_count != 0u &&
      (stride_bytes == 0u ||
       !rund::kernel::checked::mul(element_count - 1u, stride_bytes,
                                   distance) ||
       !rund::kernel::checked::add(access.offset_bytes, distance, last) ||
       !rund::kernel::checked::add(last, element_bytes, relative_end));
  if (access.resource != resource.id || element_bytes == 0u ||
      element_bytes > 8u || element_count == 0u ||
      stride_bytes < element_bytes || footprint_overflow ||
      (access.mode != AccessMode::Read && access.mode != AccessMode::Write)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  if (relative_end > resource.bytes) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  std::uint64_t physical_offset = 0u;
  std::uint64_t physical_end = 0u;
  if (!rund::kernel::checked::add(resource.alias_offset_bytes,
                                  access.offset_bytes, physical_offset) ||
      !rund::kernel::checked::add(resource.alias_offset_bytes, relative_end,
                                  physical_end)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceRangeCapacity);
  }
  return Result<PhysicalAccess>::success(PhysicalAccess{
      .access = access,
      .alias_group = resource.alias_group,
      .offset = physical_offset,
      .element_bytes = element_bytes,
      .element_count = element_count,
      .stride_bytes = stride_bytes,
      .envelope_end = physical_end,
      .subtree_begin = physical_offset,
      .subtree_end = physical_end,
      .left = NoAccess,
      .right = NoAccess,
  });
}

} // namespace

Result<bool> intersects(const Resource &left_resource, const Access &left,
                        const Resource &right_resource,
                        const Access &right) noexcept {
  auto physical_left = physical_access(left_resource, left);
  if (!physical_left) {
    return Result<bool>::fail(physical_left.reason());
  }
  auto physical_right = physical_access(right_resource, right);
  if (!physical_right) {
    return Result<bool>::fail(physical_right.reason());
  }
  return Result<bool>::success(
      find_overlap(*physical_left, *physical_right).found);
}

namespace {

[[nodiscard]] Result<Plan>
analyze_indexed(const std::span<const Resource> resources,
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
      visit_candidates(
          prior, group_roots[group], current.offset, current.envelope_end,
          candidates, stats, [&](const std::size_t before_index) {
            const PhysicalAccess &before = prior[before_index];
            if (stats != nullptr) {
              ++stats->exact_checks;
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
            const auto [row, inserted] = dependency_rows.try_emplace(
                dependency_key, plan.dependencies.size());
            if (inserted) {
              plan.dependencies.push_back(
                  Dependency{before.access.node, current.access.node});
              plan.barriers.push_back(witness);
            } else if (plan.barriers[row->second].before_resource ==
                           plan.barriers[row->second].after_resource &&
                       witness.before_resource != witness.after_resource) {
              plan.barriers[row->second] = witness;
            }
            // A complete later overlap is the visibility frontier; older edges
            // are transitive command-order mirrors, not new synchronization.
            return overlap.begin != current.offset ||
                   overlap.end != current.envelope_end;
          });
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

} // namespace

Result<Plan> analyze(const std::span<const Resource> resources,
                     const std::span<const Access> accesses,
                     const std::uint32_t node_count) {
  return analyze_indexed(resources, accesses, node_count, nullptr);
}

Result<Plan> detail::analyze_measured(const std::span<const Resource> resources,
                                      const std::span<const Access> accesses,
                                      const std::uint32_t node_count,
                                      AnalysisStats &stats) {
  stats = {};
  return analyze_indexed(resources, accesses, node_count, &stats);
}

} // namespace rund::compute::resource
