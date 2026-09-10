#include "../model.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_flow_contract {

[[nodiscard]] int CheckIndexedMap(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  const std::array<std::int64_t, 6u> source{10, 20, 30, 40, 50, 60};
  const std::array<std::uint32_t, 4u> indices{5u, 1u, 3u, 0u};
  auto flow = on(target);
  auto program = std::move(flow)
                     .input<std::int64_t>(source.size())
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
  if (!output || *output != std::vector<std::int64_t>{67, 27, 47, 17}) {
    return 3;
  }
  const std::array<std::uint32_t, 4u> invalid{5u, 6u, 3u, 0u};
  auto rejected = program->resident(source, invalid);
  if (!rejected) {
    return 4;
  }
  const Status rejected_status = rejected->run();
  if (rejected_status) {
    return 4;
  }
  if (rejected_status.reason() != Reason::GatherIndexOutOfRange) {
    return 9;
  }
  if (rejected->stats().control.overflow_ordinal != 1u) {
    return 10;
  }

  const std::array<std::uint32_t, 6u> bounded_indices{3u,  1u,  98u,
                                                      96u, 94u, 92u};
  const std::array<std::uint32_t, 6u> active_invalid{3u,  1u,  99u,
                                                     96u, 94u, 92u};
  auto bounded =
      on(target)
          .input<std::int64_t>(source.size())
          .zip_input<std::uint32_t>(bounded_indices.size())
          .branch([](auto values, auto requested) {
            auto active =
                requested.filter([](auto index) { return (index & 1u) != 0u; });
            return values.gather(active).map(
                "bounded-indexed-map", [](auto value) { return value + 7; });
          })
          .compile();
  if (!bounded) {
    return 5;
  }
  maps = 0u;
  gathers = 0u;
  for (const graph::Node &node : bounded->graph().nodes) {
    maps += node.operation == graph::Operation::Map ? 1u : 0u;
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
  }
  if (maps == 0u || gathers != 0u) {
    return 6;
  }
  auto bounded_output = bounded->run(source, bounded_indices);
  if (!bounded_output || *bounded_output != std::vector<std::int64_t>{47, 27}) {
    return 7;
  }
  auto bounded_rejected = bounded->resident(source, active_invalid);
  if (!bounded_rejected) {
    return 8;
  }
  const Status bounded_status = bounded_rejected->run();
  return !bounded_status &&
                 bounded_status.reason() == Reason::GatherIndexOutOfRange &&
                 bounded_rejected->stats().control.overflow_ordinal == 2u
             ? 0
             : 8;
}

[[nodiscard]] int CheckWideMap(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  const std::array<std::int64_t, 16u> source{
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};
  const std::array<std::uint32_t, 4u> index0{0u, 1u, 2u, 3u};
  const std::array<std::uint32_t, 4u> index1{1u, 2u, 3u, 4u};
  const std::array<std::uint32_t, 4u> index2{2u, 3u, 4u, 5u};
  const std::array<std::uint32_t, 4u> index3{3u, 4u, 5u, 6u};
  const std::array<std::uint32_t, 4u> index4{4u, 5u, 6u, 7u};
  const std::array<std::uint32_t, 4u> index5{5u, 6u, 7u, 8u};
  const std::array<std::uint32_t, 4u> index6{6u, 7u, 8u, 9u};
  const std::array<std::uint32_t, 4u> index7{7u, 8u, 9u, 10u};
  const std::array<std::uint32_t, 4u> index8{8u, 9u, 10u, 11u};

  auto program =
      on(target)
          .input<std::int64_t>(source.size())
          .zip_input<std::uint32_t>(index0.size())
          .zip_input<std::uint32_t>(index1.size())
          .zip_input<std::uint32_t>(index2.size())
          .zip_input<std::uint32_t>(index3.size())
          .zip_input<std::uint32_t>(index4.size())
          .zip_input<std::uint32_t>(index5.size())
          .zip_input<std::uint32_t>(index6.size())
          .zip_input<std::uint32_t>(index7.size())
          .zip_input<std::uint32_t>(index8.size())
          .branch([](auto values, auto i0, auto i1, auto i2, auto i3, auto i4,
                     auto i5, auto i6, auto i7, auto i8) {
            return zip(values.gather(i0), values.gather(i1), values.gather(i2),
                       values.gather(i3), values.gather(i4), values.gather(i5),
                       values.gather(i6), values.gather(i7), values.gather(i8))
                .map("wide-indexed-map",
                     [](auto v0, auto v1, auto v2, auto v3, auto v4, auto v5,
                        auto v6, auto v7, auto v8) {
                       return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
                     });
          })
          .compile();
  if (!program || program->graph().nodes.size() != 1u ||
      program->graph().nodes.front().operation != graph::Operation::Map ||
      program->graph().nodes.front().accesses.size() != 19u) {
    return 1;
  }
  auto output = program->run(source, index0, index1, index2, index3, index4,
                             index5, index6, index7, index8);
  if (!output || *output != std::vector<std::int64_t>{450, 540, 630, 720}) {
    return 2;
  }
  auto bad0 = index0;
  auto bad1 = index1;
  bad0[3u] = static_cast<std::uint32_t>(source.size());
  bad1[1u] = static_cast<std::uint32_t>(source.size());
  auto rejected = program->resident(source, bad0, bad1, index2, index3, index4,
                                    index5, index6, index7, index8);
  if (!rejected) {
    return 3;
  }
  const Status status = rejected->run();
  return !status && status.reason() == Reason::GatherIndexOutOfRange &&
                 rejected->stats().control.overflow_ordinal == 1u
             ? 0
             : 4;
}

[[nodiscard]] int CheckIndexedCapacity() {
  using namespace rund::compute;
  static_assert(std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<std::uint32_t>::max());
  constexpr std::size_t Count =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1u;
  auto rejected =
      on(Target::cpu(1u))
          .input<std::int64_t>(1u)
          .zip_input<std::uint32_t>(Count)
          .branch([](auto values, auto indices) {
            return values.gather(indices).map(
                "indexed-capacity", [](auto value) { return value + 1; });
          })
          .compile();
  return !rejected && rejected.reason() == Reason::GraphInvalid ? 0 : 1;
}

} // namespace rund_node_flow_contract
