#include "../model.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_flow_contract {

[[nodiscard]] int CheckIndexedAlias(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  std::array<std::int64_t, 64u> source{};
  std::array<std::uint32_t, 64u> previous{};
  for (std::size_t index = 0u; index < source.size(); ++index) {
    source[index] = static_cast<std::int64_t>(index);
  }

  auto program =
      on(target)
          .input<std::int64_t>(source.size())
          .zip_input<std::uint32_t>(previous.size())
          .branch([](auto input, auto indices) {
            auto staged = input.map("alias-source",
                                    [](auto value) { return value + 100; });
            return zip(staged, staged.gather(indices))
                .map("alias-read",
                     [](auto direct, auto indexed) { return direct - indexed; })
                .reduce(Reduce::Sum);
          })
          .compile();
  if (!program) {
    return 1;
  }

  std::size_t maps = 0u;
  std::size_t gathers = 0u;
  std::size_t reductions = 0u;
  bool indexed_output = false;
  for (const graph::Node &node : program->graph().nodes) {
    maps += node.operation == graph::Operation::Map ? 1u : 0u;
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
    reductions += node.operation == graph::Operation::Reduce ? 1u : 0u;
    if (node.operation != graph::Operation::Map || node.accesses.size() != 4u) {
      continue;
    }
    const auto output =
        std::find_if(node.accesses.begin(), node.accesses.end(),
                     [](const graph::Access access) {
                       return access.mode == resource::AccessMode::Write;
                     });
    if (output != node.accesses.end() &&
        output->resource <= program->graph().resources.size()) {
      indexed_output = true;
      if (program->graph().resources[output->resource - 1u].source != 0u) {
        return 2;
      }
    }
  }
  if (maps != 2u || gathers != 0u || reductions != 1u || !indexed_output) {
    return 3;
  }
  auto output = program->run(source, previous);
  return output &&
                 *output == std::vector<std::int64_t>{static_cast<std::int64_t>(
                                source.size() * (source.size() - 1u) / 2u)}
             ? 0
             : 4;
}

} // namespace rund_node_flow_contract
