#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_graph_services::memory_detail {

bool CheckRejectedMemoryPlan() {
  Info read_before_reset{};
  read_before_reset.resources = {
      ResourceOf(1u, Value::U32, 4u, 4u),
  };
  read_before_reset.resources.front().visibility = Visibility::Output;
  const Resource &late_output = read_before_reset.resources.front();
  read_before_reset.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = late_output.elements,
           .accesses = {AccessOf(late_output, AccessMode::Read)}},
      Node{.index = 1u,
           .operation = Operation::Compact,
           .elements = late_output.elements,
           .accesses = {AccessOf(late_output, AccessMode::Write)}},
  };
  const std::array<MemoryNode, 2u> late_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Partial},
  };
  const auto late_rejected = Plan(read_before_reset, late_writes);
  if (late_rejected ||
      late_rejected.reason() != rund::compute::Reason::GraphInvalid) {
    return false;
  }

  constexpr std::uint64_t widest =
      std::numeric_limits<std::uint64_t>::max() / 4u;
  Info overflow{};
  overflow.resources = {
      ResourceOf(1u, Value::U32, widest, 4u),
      ResourceOf(2u, Value::U32, widest, 4u),
  };
  const Resource &overflow_left = overflow.resources[0u];
  const Resource &overflow_right = overflow.resources[1u];
  overflow.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = widest,
           .accesses = {AccessOf(overflow_left, AccessMode::Write),
                        AccessOf(overflow_right, AccessMode::Write)}},
  };
  const std::array<MemoryNode, 1u> overflow_writes{
      MemoryNode{.write = Write::Full}};
  const auto rejected = Plan(overflow, overflow_writes, WidePage);
  return !rejected && rejected.reason() == rund::compute::Reason::GraphCapacity;
}

} // namespace rund_node_graph_services::memory_detail
