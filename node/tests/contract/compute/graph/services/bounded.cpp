#include "model.hpp"

#include <array>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] int CheckBounded(rund::compute::Device *const device,
                               rund::compute::ProgramCache *const cache) {
  const std::array<std::int32_t, 4> low{1, 2, 3, 4};
  auto bounded = rund::compute::on(*device, *cache)
                     .map<std::int32_t>("bounded-source", 4u,
                                        [](auto value) { return value; })
                     .filter([](auto value) { return value > 2; })
                     .compile();
  if (!bounded || !ValidResourceGraph(bounded->graph())) {
    return 11;
  }
  const Info bounded_graph = bounded->graph();
  if (bounded_graph.nodes.size() != 3u ||
      bounded_graph.nodes[0u].operation != Operation::Map ||
      bounded_graph.nodes[0u].accesses.size() != 3u ||
      bounded_graph.nodes[1u].operation != Operation::Partition ||
      bounded_graph.nodes[2u].operation != Operation::Reduce) {
    return 51;
  }
  auto bounded_job = bounded->resident(std::span<const std::int32_t>{low});
  if (!bounded_job || !bounded_job->run()) {
    return 12;
  }
  auto bounded_values = bounded_job->read();
  if (!bounded_values || *bounded_values != std::vector<std::int32_t>{3, 4}) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_graph_services
