#pragma once

#include "../model.hpp"

#include "../../../../../../src/compute/resource/memory.hpp"

#include <cstdint>
#include <span>

namespace rund_node_graph_services::memory_detail {

using rund::compute::detail::resource_detail::Inplace;
using rund::compute::detail::resource_detail::MemoryNode;
using rund::compute::detail::resource_detail::Write;
using rund::compute::graph::Access;
using rund::compute::graph::Info;
using rund::compute::graph::Node;
using rund::compute::graph::Resource;
using rund::compute::graph::Value;

inline constexpr std::uint64_t WidePage = 4ull << 30u;

[[nodiscard]] Resource ResourceOf(std::uint32_t id, Value type,
                                  std::uint64_t elements,
                                  std::uint64_t width) noexcept;
[[nodiscard]] Access AccessOf(const Resource &value, AccessMode mode) noexcept;
[[nodiscard]] bool Overlaps(const Resource &left,
                            const Resource &right) noexcept;
[[nodiscard]] bool SamePlan(const Info &left, const Info &right) noexcept;
[[nodiscard]] rund::compute::Status
Plan(Info &info, std::span<const MemoryNode> nodes,
     std::uint64_t page = 1ull << 30u) noexcept;

[[nodiscard]] bool CheckBasicMemoryPlan();
[[nodiscard]] bool CheckWideMemoryPlan();
[[nodiscard]] bool CheckAliasMemoryPlan();
[[nodiscard]] bool CheckCapacityMemoryPlan();
[[nodiscard]] bool CheckRejectedMemoryPlan();

} // namespace rund_node_graph_services::memory_detail
