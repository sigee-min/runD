#include "local.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckBoundedGather() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> source{10, 20, 30, 40};
  const std::array<std::uint32_t, 4u> indices{2u, 99u, 1u, 0u};
  auto program = on(Target::cpu(2u))
                     .input<std::int32_t>(source.size())
                     .zip_input<std::uint32_t>(indices.size())
                     .branch([](auto values, auto requested) {
                       auto active = requested.filter([](auto index) {
                         return index != std::uint32_t{99};
                       });
                       return values.gather(active);
                     })
                     .compile();
  if (!program) {
    return 1;
  }
  const auto backend = program->backend();
  if (!backend || *backend != Backend::Cpu)
    return 1;
  const auto fingerprint = program->graph().fingerprint;
  const std::size_t nodes = program->graph().nodes.size();
  const std::size_t resources = program->graph().resources.size();
  auto gathered = program->run(source, indices);
  if (!gathered || *gathered != std::vector<std::int32_t>{30, 20, 10}) {
    return 2;
  }
  const std::array<std::uint32_t, 4u> invalid{2u, 9u, 1u, 0u};
  auto rejected = program->run(source, invalid);
  return !rejected && rejected.error() == "compute_gather_index_out_of_range" &&
                 *backend == Backend::Cpu &&
                 program->graph().fingerprint == fingerprint &&
                 program->graph().nodes.size() == nodes &&
                 program->graph().resources.size() == resources
             ? 0
             : 3;
}

[[nodiscard]] int CheckIndexedMap() {
  using namespace rund::compute;
  const std::array<std::int32_t, 6u> source{10, 20, 30, 40, 50, 60};
  const std::array<std::uint32_t, 4u> indices{5u, 1u, 3u, 0u};
  auto program = on(Target::cpu(2u))
                     .input<std::int32_t>(source.size())
                     .zip_input<std::uint32_t>(indices.size())
                     .branch([](auto values, auto requested) {
                       return values.gather(requested).map(
                           "indexed-map", [](auto value) { return value + 7; });
                     })
                     .compile();
  if (!program) {
    return 1;
  }
  std::size_t maps = 0u;
  std::size_t gathers = 0u;
  for (const graph::Node &node : program->graph().nodes) {
    maps += node.operation == graph::Operation::Map ? 1u : 0u;
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
  }
  if (maps != 1u || gathers != 0u || program->graph().nodes.size() != 1u) {
    return 2;
  }
  auto output = program->run(source, indices);
  if (!output || *output != std::vector<std::int32_t>{67, 27, 47, 17}) {
    return 3;
  }
  const std::array<std::uint32_t, 4u> invalid{5u, 6u, 3u, 0u};
  auto rejected = program->run(source, invalid);
  return !rejected && rejected.error() == "compute_gather_index_out_of_range"
             ? 0
             : 4;
}

} // namespace rund_node_test_flow_primitives
