#include "model.hpp"

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

[[nodiscard]] int CheckFusionBoundaries(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  using Source = Fixed<16, 16>;
  using Middle = Fixed<8, 24>;
  constexpr std::array<Source, 3u> fixed_input{
      Source::from_raw(1 << 16),
      Source::from_raw(2 << 16),
      Source::from_raw(3 << 16),
  };
  auto fixed = on(target)
                   .input<Source>(fixed_input.size())
                   .map("fixed-format-enter",
                        [](auto value) { return quantize<Middle>(value); })
                   .map("fixed-format-leave",
                        [](auto value) { return quantize<Source>(value); })
                   .compile();
  if (!fixed) {
    return 1;
  }
  const std::size_t fixed_maps = static_cast<std::size_t>(
      std::count_if(fixed->graph().nodes.begin(), fixed->graph().nodes.end(),
                    [](const graph::Node &node) {
                      return node.operation == graph::Operation::Map;
                    }));
  auto fixed_output = fixed->run(fixed_input);
  if (fixed_maps != 2u || !fixed_output ||
      *fixed_output !=
          std::vector<Source>(fixed_input.begin(), fixed_input.end())) {
    return 2;
  }

  constexpr std::array<std::int32_t, 4u> controlled_input{1, 2, 0, 3};
  auto controlled =
      on(target)
          .input<std::int32_t>(controlled_input.size())
          .branch([](auto values) {
            auto active = values.filter([](auto value) { return value > 0; });
            return active.template unroll<2u>(
                [](auto work) {
                  return work.map("controlled-fusion-boundary",
                                  [](auto value) { return value + 1; });
                },
                [](auto value) { return value == 999; });
          })
          .compile();
  if (!controlled) {
    return 3;
  }
  const auto &controlled_graph = controlled->graph();
  const std::size_t controlled_maps = static_cast<std::size_t>(std::count_if(
      controlled_graph.nodes.begin(), controlled_graph.nodes.end(),
      [&](const graph::Node &node) {
        if (node.operation != graph::Operation::Map) {
          return false;
        }
        bool reads_value = false;
        bool writes_value = false;
        for (const graph::Access access : node.accesses) {
          if (access.resource == 0u ||
              access.resource > controlled_graph.resources.size() ||
              controlled_graph.resources[access.resource - 1u].type !=
                  graph::Value::I32) {
            continue;
          }
          reads_value =
              reads_value || access.mode == resource::AccessMode::Read;
          writes_value =
              writes_value || access.mode == resource::AccessMode::Write;
        }
        return reads_value && writes_value;
      }));
  auto controlled_output = controlled->run(controlled_input);
  if (controlled_maps != 2u || !controlled_output ||
      *controlled_output != std::vector<std::int32_t>{3, 4, 5}) {
    std::fprintf(stderr,
                 "fusion boundary backend=%u maps=%zu output=%u count=%zu\n",
                 static_cast<unsigned>(backend), controlled_maps,
                 controlled_output ? 1u : 0u,
                 controlled_output ? controlled_output->size() : 0u);
    return 4;
  }
  return 0;
}

[[nodiscard]] int CheckResetProjection(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  constexpr std::uint32_t Convex = 3u;
  constexpr std::uint32_t Mesh = 2u;
  constexpr std::uint64_t Width = Convex + Mesh;
  constexpr std::size_t OutputCount = 2u * Width;
  constexpr std::array<std::uint64_t, 2u * Convex> convex{11u,  0u, 0u,
                                                          101u, 0u, 0u};
  constexpr std::array<std::uint64_t, 2u * Mesh> mesh{17u, 0u, 107u, 0u};
  const std::vector<std::uint64_t> expected{11u,  0u, 0u, 17u,  0u,
                                            101u, 0u, 0u, 107u, 0u};

  auto program =
      on(target)
          .input<std::uint64_t>(convex.size())
          .zip_input<std::uint64_t>(mesh.size())
          .branch([](auto left, auto right) {
            const auto left_ordinal =
                left.indices()
                    .map("reset-projection",
                         [](auto index) { return mask(index == index); })
                    .scan(Scan::ExclusiveSum);
            const auto right_ordinal =
                right.indices()
                    .map("reset-projection",
                         [](auto index) { return mask(index == index); })
                    .scan(Scan::ExclusiveSum);
            const auto left_target = left_ordinal.map(
                "reset-projection",
                capture(
                    [](auto index, auto source, auto output) {
                      const auto field = index / source;
                      return field * output + index - field * source;
                    },
                    Convex, static_cast<std::uint32_t>(Width)));
            const auto right_target = right_ordinal.map(
                "reset-projection",
                capture(
                    [](auto index, auto source, auto output, auto offset) {
                      const auto field = index / source;
                      return field * output + offset + index - field * source;
                    },
                    Mesh, static_cast<std::uint32_t>(Width), Convex));
            const auto left_segment =
                left.scatter(left_target, {.count = OutputCount});
            const auto right_segment =
                right.scatter(right_target, {.count = OutputCount});
            return zip(left_segment, right_segment, left_segment.indices())
                .map("reset-projection",
                     capture(
                         [](auto left_value, auto right_value, auto index,
                            auto width, auto left_count) {
                           const auto field = index / width;
                           return select(index - field * width < left_count,
                                         left_value, right_value);
                         },
                         Width, static_cast<std::uint64_t>(Convex)));
          })
          .compile();
  if (!program || program->graph().memory.reset_count != 2u) {
    return 1;
  }
  auto job = program->resident(std::span{convex}, std::span{mesh});
  if (!job || !job->run()) {
    return 2;
  }
  const auto first = job->read();
  const auto first_stats = job->stats();
  const bool fusion_required = backend != Backend::Cpu;
  if (!first || *first != expected ||
      (fusion_required && first_stats.fusions == 0u) || !job->run()) {
    return 3;
  }
  const auto warm = job->read();
  return warm && *warm == expected &&
                 (!fusion_required || job->stats().fusions > 0u)
             ? 0
             : 4;
}

[[nodiscard]] int
CheckBackendContracts(const std::span<const rund::compute::Backend> backends) {
  if (CheckIndexedCapacity() != 0) {
    return 39;
  }
  std::uint64_t expression_graph_hash = 0u;
  std::uint64_t expression_output_hash = 0u;
  std::array<FlowHash, 6u> composition_reference{};
  std::array<FlowHash, 3u> typed_reference{};
  for (const rund::compute::Backend backend : backends) {
    if (const int result = CheckIndexedMap(backend); result != 0) {
      return 40 + result;
    }
    if (const int result = CheckWideMap(backend); result != 0) {
      return 45 + result;
    }
    if (const int result = CheckIndexedAlias(backend); result != 0) {
      return 48 + result;
    }
    if (const int result = CheckFusionBoundaries(backend); result != 0) {
      return 52 + result;
    }
    if (const int result = CheckResetProjection(backend); result != 0) {
      return 56 + result;
    }
    if (const int result = CheckExpressions(backend, expression_graph_hash,
                                            expression_output_hash);
        result != 0) {
      return 50 + result;
    }
    if (const int result = CheckRecords(backend); result != 0) {
      return 60 + result;
    }
    if (!CheckComposition(backend, composition_reference)) {
      return 70 + static_cast<int>(backend);
    }
    if (!CheckTyped(backend, typed_reference)) {
      return 80 + static_cast<int>(backend);
    }
  }
  return 0;
}

} // namespace rund_node_flow_contract
