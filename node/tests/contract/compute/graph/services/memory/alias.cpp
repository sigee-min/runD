#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_graph_services::memory_detail {

bool CheckAliasMemoryPlan() {
  Info multi_source{};
  multi_source.resources = {
      ResourceOf(1u, Value::U32, 128u, 4u),
      ResourceOf(2u, Value::U32, 128u, 4u),
      ResourceOf(3u, Value::U32, 128u, 4u),
  };
  const Resource &left_source = multi_source.resources[0u];
  const Resource &right_source = multi_source.resources[1u];
  const Resource &map_output = multi_source.resources[2u];
  multi_source.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = left_source.elements,
           .accesses = {AccessOf(left_source, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = right_source.elements,
           .accesses = {AccessOf(right_source, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = map_output.elements,
           .accesses = {AccessOf(left_source, AccessMode::Read),
                        AccessOf(right_source, AccessMode::Read),
                        AccessOf(map_output, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = map_output.elements,
           .accesses = {AccessOf(map_output, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> multi_source_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::Pointwise},
      MemoryNode{.write = Write::Full},
  };
  Info indexed_source = multi_source;
  const std::array<MemoryNode, 4u> indexed_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::None},
      MemoryNode{.write = Write::Full},
  };
  if (!Plan(multi_source, multi_source_writes) ||
      multi_source.memory.logical_bytes != 1536u ||
      multi_source.memory.live_bytes != 1536u ||
      multi_source.memory.physical_bytes != 1024u ||
      multi_source.memory.allocation_count != 1u ||
      multi_source.resources[2u].source != 1u ||
      multi_source.resources[2u].alias_group !=
          multi_source.resources[0u].alias_group ||
      multi_source.resources[2u].alias_offset_bytes !=
          multi_source.resources[0u].alias_offset_bytes ||
      Overlaps(multi_source.resources[0u], multi_source.resources[1u])) {
    return false;
  }
  if (!Plan(indexed_source, indexed_writes) ||
      indexed_source.memory.logical_bytes != 1536u ||
      indexed_source.memory.live_bytes != 1536u ||
      indexed_source.memory.physical_bytes != 1536u ||
      indexed_source.memory.allocation_count != 1u ||
      indexed_source.resources[2u].source != 0u ||
      Overlaps(indexed_source.resources[0u], indexed_source.resources[2u])) {
    return false;
  }
  return true;
}

} // namespace rund_node_graph_services::memory_detail
