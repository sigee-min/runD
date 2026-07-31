#include "local.hpp"

#include "../../../../../src/compute/resource/memory.hpp"

#include <array>

namespace rund_node_graph_services {

[[nodiscard]] bool ValidMemoryBirth() {
  using rund::compute::graph::Access;
  using rund::compute::graph::Info;
  using rund::compute::graph::Node;
  using rund::compute::graph::Resource;
  using rund::compute::resource::AccessMode;

  const auto resource = Resource{.id = 1u,
                                 .type = rund::compute::graph::Value::U32,
                                 .visibility = Visibility::Internal,
                                 .elements = 4u,
                                 .element_bytes = 4u,
                                 .bytes = 16u};
  const auto access = [](const AccessMode mode) {
    return Access{.resource = 1u,
                  .mode = mode,
                  .size_bytes = 16u,
                  .element_bytes = 4u,
                  .element_count = 4u,
                  .stride_bytes = 4u};
  };

  Info invalid{};
  invalid.resources.push_back(resource);
  invalid.nodes.push_back(
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = 4u,
           .accesses = {access(AccessMode::Read), access(AccessMode::Write)}});
  using rund::compute::detail::resource_detail::MemoryNode;
  using rund::compute::detail::resource_detail::Write;
  const auto plan = [](Info &info,
                       const std::span<const MemoryNode> nodes) {
    return rund::compute::detail::resource_detail::plan_memory(
        info, nodes, 1ull << 30u);
  };
  const std::array<MemoryNode, 1u> invalid_proof{
      MemoryNode{.write = Write::Full}};
  const auto rejected = plan(
      invalid, invalid_proof);
  if (rejected || rejected.reason() != rund::compute::Reason::GraphInvalid) {
    return false;
  }

  Info partial{};
  partial.resources.push_back(resource);
  auto partial_write = access(AccessMode::Write);
  partial_write.size_bytes = 12u;
  partial_write.element_count = 3u;
  partial.nodes.push_back(Node{.index = 0u,
                               .operation = Operation::Map,
                               .elements = 3u,
                               .accesses = {partial_write}});
  partial.nodes.push_back(Node{.index = 1u,
                               .operation = Operation::Map,
                               .elements = 4u,
                               .accesses = {access(AccessMode::Read)}});
  const std::array<MemoryNode, 2u> partial_proof{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  const auto partial_rejected =
      plan(partial,
                                                          partial_proof);
  if (partial_rejected ||
      partial_rejected.reason() != rund::compute::Reason::GraphInvalid) {
    return false;
  }

  Info valid{};
  valid.resources.push_back(resource);
  valid.nodes.push_back(Node{.index = 0u,
                             .operation = Operation::Map,
                             .elements = 4u,
                             .accesses = {access(AccessMode::Write)}});
  valid.nodes.push_back(Node{.index = 1u,
                             .operation = Operation::Map,
                             .elements = 4u,
                             .accesses = {access(AccessMode::Read)}});
  Info missing_proof = valid;
  const auto missing = plan(
      missing_proof, std::span<const MemoryNode>{});
  if (missing || missing.reason() != rund::compute::Reason::GraphInvalid) {
    return false;
  }
  const std::array<MemoryNode, 2u> valid_proof{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  const auto accepted =
      plan(valid, valid_proof);
  if (!accepted || valid.resources.front().requires_reset() ||
      valid.memory.logical_bytes != 16u || valid.memory.live_bytes != 16u ||
      valid.memory.physical_bytes != 16u ||
      valid.memory.allocation_count != 1u || valid.memory.reset_bytes != 0u ||
      valid.memory.reset_count != 0u) {
    return false;
  }

  Info conditional = valid;
  const std::array<MemoryNode, 2u> complete_writes{
      MemoryNode{.write = Write::Partial}, MemoryNode{.write = Write::Full}};
  const auto initialized = plan(
      conditional, complete_writes);
  if (!initialized || !conditional.resources.front().requires_reset() ||
      conditional.resources.front().alias_group !=
          conditional.resources.front().id ||
      conditional.memory.logical_bytes != 16u ||
      conditional.memory.live_bytes != 16u ||
      conditional.memory.physical_bytes != 16u ||
      conditional.memory.allocation_count != 1u ||
      conditional.memory.reset_bytes != 16u ||
      conditional.memory.reset_count != 1u) {
    return false;
  }

  using rund::compute::detail::resource_detail::Domain;
  Info guarded = valid;
  guarded.resources.push_back(Resource{.id = 2u,
                                       .type = rund::compute::graph::Value::U32,
                                       .visibility = Visibility::Input,
                                       .elements = 1u,
                                       .element_bytes = 4u,
                                       .bytes = 4u});
  guarded.resources.push_back(Resource{.id = 3u,
                                       .type = rund::compute::graph::Value::U32,
                                       .visibility = Visibility::Input,
                                       .elements = 1u,
                                       .element_bytes = 4u,
                                       .bytes = 4u,
                                       .parent = 2u});
  const Domain active{.count = 2u};
  const std::array<MemoryNode, 2u> guarded_writes{
      MemoryNode{.domain = active, .write = Write::Domain},
      MemoryNode{.domain = Domain{.count = 3u}, .write = Write::Partial}};
  if (!plan(guarded,
                                                           guarded_writes) ||
      guarded.resources.front().requires_reset()) {
    return false;
  }

  Info escaped = guarded;
  escaped.resources[2u].parent = 0u;
  auto escaped_writes = guarded_writes;
  if (!plan(escaped,
                                                           escaped_writes) ||
      !escaped.resources.front().requires_reset()) {
    return false;
  }

  Info external = valid;
  external.resources.front().visibility = Visibility::Output;
  if (!plan(external,
                                                           complete_writes) ||
      !external.resources.front().requires_reset() ||
      external.resources.front().reset_node != 0u ||
      external.memory.logical_bytes != 0u ||
      external.memory.physical_bytes != 0u ||
      external.memory.allocation_count != 0u ||
      external.memory.reset_bytes != 16u || external.memory.reset_count != 1u) {
    return false;
  }

  Info cycle = guarded;
  cycle.resources[1u].parent = 3u;
  const auto rejected_cycle =
      plan(cycle,
                                                          guarded_writes);
  return !rejected_cycle &&
         rejected_cycle.reason() == rund::compute::Reason::GraphInvalid;
}

} // namespace rund_node_graph_services
