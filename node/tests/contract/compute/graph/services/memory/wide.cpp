#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_graph_services::memory_detail {

bool CheckWideMemoryPlan() {
  constexpr std::uint64_t MiB = 1ull << 20u;
  constexpr std::uint64_t large_bytes = 700u * MiB;
  Info chunked{};
  chunked.resources = {
      ResourceOf(1u, Value::U32, large_bytes / 4u, 4u),
      ResourceOf(2u, Value::U32, large_bytes / 4u, 4u),
  };
  const Resource &chunk_left = chunked.resources[0u];
  const Resource &chunk_right = chunked.resources[1u];
  chunked.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = chunk_left.elements,
           .accesses = {AccessOf(chunk_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = chunk_right.elements,
           .accesses = {AccessOf(chunk_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = chunk_left.elements,
           .accesses = {AccessOf(chunk_left, AccessMode::Read),
                        AccessOf(chunk_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> chunk_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
  };
  if (!Plan(chunked, chunk_writes, WidePage) ||
      chunked.memory.logical_bytes != 2u * large_bytes ||
      chunked.memory.live_bytes != 2u * large_bytes ||
      chunked.memory.physical_bytes != 2u * large_bytes ||
      chunked.memory.allocation_count != 2u ||
      chunked.resources[0u].alias_group == chunked.resources[1u].alias_group ||
      chunked.resources[0u].alias_offset_bytes != 0u ||
      chunked.resources[1u].alias_offset_bytes != 0u) {
    return false;
  }
  return true;
}

} // namespace rund_node_graph_services::memory_detail
