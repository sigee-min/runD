#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_graph_services::memory_detail {

bool CheckBasicMemoryPlan() {
  Info authored{};
  authored.resources = {
      ResourceOf(1u, Value::U32, 128u, 4u),
      ResourceOf(2u, Value::U64, 32u, 8u),
      ResourceOf(3u, Value::U32, 32u, 4u),
  };
  const Resource &wide32 = authored.resources[0u];
  const Resource &wide64 = authored.resources[1u];
  const Resource &small32 = authored.resources[2u];
  authored.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = wide32.elements,
           .accesses = {AccessOf(wide32, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = wide32.elements,
           .accesses = {AccessOf(wide32, AccessMode::Read),
                        AccessOf(wide64, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = wide64.elements,
           .accesses = {AccessOf(wide64, AccessMode::Read),
                        AccessOf(small32, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = small32.elements,
           .accesses = {AccessOf(small32, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> complete{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  Info planned = authored;
  Info repeated = authored;
  const auto first = Plan(planned, complete);
  const auto second = Plan(repeated, complete);
  if (!first || !second || !SamePlan(planned, repeated) ||
      planned.memory.logical_bytes != 896u ||
      planned.memory.live_bytes != 768u ||
      planned.memory.physical_bytes != 768u ||
      planned.memory.allocation_count != 1u) {
    return false;
  }
  const Resource &first32 = planned.resources[0u];
  const Resource &first64 = planned.resources[1u];
  const Resource &last32 = planned.resources[2u];
  if (first32.alias_group != first64.alias_group ||
      first32.alias_group != last32.alias_group ||
      first32.alias_offset_bytes != last32.alias_offset_bytes ||
      first32.bytes == last32.bytes || Overlaps(first32, first64) ||
      Overlaps(first64, last32) || first32.type != Value::U32 ||
      first64.type != Value::U64) {
    return false;
  }
  for (const Resource &value : planned.resources) {
    if (value.alias_offset_bytes % 256u != 0u || value.requires_reset()) {
      return false;
    }
  }

  Info reset{};
  reset.resources = {
      ResourceOf(1u, Value::U32, 16u, 4u),
      ResourceOf(2u, Value::U64, 4u, 8u),
  };
  const Resource &reset32 = reset.resources[0u];
  const Resource &reset64 = reset.resources[1u];
  reset.nodes = {
      Node{.index = 0u,
           .operation = Operation::Compact,
           .elements = reset32.elements,
           .accesses = {AccessOf(reset32, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = reset32.elements,
           .accesses = {AccessOf(reset32, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Compact,
           .elements = reset64.elements,
           .accesses = {AccessOf(reset64, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = reset64.elements,
           .accesses = {AccessOf(reset64, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> partial{
      MemoryNode{.write = Write::Partial}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Partial}, MemoryNode{.write = Write::Full}};
  if (!Plan(reset, partial) || reset.memory.logical_bytes != 96u ||
      reset.memory.live_bytes != 64u || reset.memory.physical_bytes != 64u ||
      reset.memory.allocation_count != 1u || reset.memory.reset_bytes != 96u ||
      reset.memory.reset_count != 2u) {
    return false;
  }
  const Resource &stored32 = reset.resources[0u];
  const Resource &stored64 = reset.resources[1u];
  if (!stored32.requires_reset() || !stored64.requires_reset() ||
      stored32.alias_group != stored64.alias_group ||
      stored32.alias_offset_bytes != 0u || stored64.alias_offset_bytes != 0u ||
      !Overlaps(stored32, stored64)) {
    return false;
  }
  return true;
}

} // namespace rund_node_graph_services::memory_detail
