#include "model.hpp"

#include <algorithm>

namespace rund::compute::resource::plan_detail {
namespace {

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

[[nodiscard]] bool older(const Candidate left, const Candidate right) noexcept {
  return left.latest < right.latest;
}

} // namespace

const Resource *find_resource(const std::span<const Resource> resources,
                              const std::uint32_t id) noexcept {
  return id == 0u || id > resources.size() ? nullptr : &resources[id - 1u];
}

std::size_t insert_access(std::vector<PhysicalAccess> &accesses,
                          const std::size_t root, const std::size_t inserted,
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

void visit_candidates(const std::vector<PhysicalAccess> &accesses,
                      const std::size_t root, const std::uint64_t begin,
                      const std::uint64_t end, std::vector<Candidate> &queue,
                      detail::AnalysisStats *const stats,
                      const CandidateVisitor visit, void *const context) {
  const auto push_subtree = [&](const std::size_t index) {
    if (index == NoAccess) {
      return;
    }
    const PhysicalAccess &access = accesses[index];
    if (access.subtree_end <= begin || access.subtree_begin >= end) {
      return;
    }
    queue.push_back(Candidate{.root = index, .latest = access.subtree_latest});
    std::push_heap(queue.begin(), queue.end(), older);
  };
  const auto push_single = [&](const std::size_t index) {
    const PhysicalAccess &access = accesses[index];
    if (access.envelope_end <= begin || access.offset >= end) {
      return;
    }
    queue.push_back(Candidate{.root = index, .latest = index, .single = true});
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
      if (!visit(context, current.root)) {
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

} // namespace rund::compute::resource::plan_detail
