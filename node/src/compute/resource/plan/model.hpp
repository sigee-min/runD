#pragma once

#include "../index.hpp"

#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace rund::compute::resource::plan_detail {

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

struct Candidate final {
  std::size_t root{};
  std::size_t latest{};
  bool single{};
};

using CandidateVisitor = bool (*)(void *context, std::size_t index);

[[nodiscard]] const Resource *find_resource(std::span<const Resource> resources,
                                            std::uint32_t id) noexcept;

[[nodiscard]] Result<PhysicalAccess>
physical_access(const Resource &resource, const Access &access) noexcept;

[[nodiscard]] Overlap find_overlap(const PhysicalAccess &left,
                                   const PhysicalAccess &right) noexcept;

[[nodiscard]] std::size_t insert_access(std::vector<PhysicalAccess> &accesses,
                                        std::size_t root, std::size_t inserted,
                                        detail::AnalysisStats *stats) noexcept;

void visit_candidates(const std::vector<PhysicalAccess> &accesses,
                      std::size_t root, std::uint64_t begin, std::uint64_t end,
                      std::vector<Candidate> &queue,
                      detail::AnalysisStats *stats, CandidateVisitor visit,
                      void *context);

[[nodiscard]] Result<Plan> analyze_indexed(std::span<const Resource> resources,
                                           std::span<const Access> accesses,
                                           std::uint32_t node_count,
                                           detail::AnalysisStats *stats);

} // namespace rund::compute::resource::plan_detail
