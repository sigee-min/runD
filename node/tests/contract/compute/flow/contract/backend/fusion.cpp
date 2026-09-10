#include "../../../../../../src/compute/flow/state.hpp"
#include "../model.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_flow_contract {

namespace {

[[nodiscard]] bool CheckFilterDescriptorDomain() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  const std::uint64_t limit = std::numeric_limits<std::uint32_t>::max();
  for (const Type type : {Type::U32, Type::U64}) {
    for (const std::uint64_t capacity : {limit, limit + 1u}) {
      // A narrow host cannot author the descriptor above its size_t domain.
      if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (capacity > std::numeric_limits<std::size_t>::max())
          continue;
      }
      const auto count = static_cast<std::size_t>(capacity);
      const auto flow = make_flow(Target::cpu(), type, count);
      const auto expression = make_expr();
      const auto value = detail::input(expression, type, 0u);
      const auto zero = constant(expression, type, 0u);
      const auto filtered =
          flow_filter_value(flow, 1u, binary(ExprOp::NotEqual, value, zero),
                            binary(ExprOp::Equal, value, zero));
      const auto output = flow_bounded_reduce_value(
          flow, filtered.values, filtered.count, Reduce::Sum);
      if (!flow || !flow->status || output == 0u) {
        return false;
      }
      const auto *reduce = std::get_if<FlowPrimitive>(&flow->steps.back());
      if (reduce == nullptr || reduce->operation != Primitive::Reduce) {
        return false;
      }
      const auto inputs = flow->value_ids.view(reduce->inputs);
      if (count == limit) {
        if (inputs.size() != 1u)
          return false;
      } else if (inputs.size() != 2u || inputs[0u] != filtered.values ||
                 inputs[1u] != filtered.count) {
        return false;
      }
    }
  }
  return true;
}

template <class T>
[[nodiscard]] int CheckFilteredSum(const rund::compute::Target target) {
  using namespace rund::compute;
  const auto operations = [](const auto &program, const graph::Operation op) {
    return std::count_if(
        program.graph().nodes.begin(), program.graph().nodes.end(),
        [op](const auto &node) { return node.operation == op; });
  };
  const auto selected = [](const T value) { return (value & T{7}) != T{0}; };
  for (const std::size_t count : {0u, 1u, 257u, 4097u}) {
    std::vector<T> input(count);
    T expected{};
    for (std::size_t i = 0u; i < count; ++i) {
      input[i] = static_cast<T>(i);
      if (selected(input[i]))
        expected += input[i];
    }
    auto sum = on(target)
                   .template map<T>("filter-source", count,
                                    [](auto value) { return value; })
                   .filter([](auto value) { return (value & T{7}) != T{0}; })
                   .reduce(Reduce::Sum)
                   .compile();
    if (!sum || operations(*sum, graph::Operation::Partition) != 0 ||
        operations(*sum, graph::Operation::Reduce) != (count == 0u ? 0 : 1)) {
      std::fprintf(stderr,
                   "sum compile count=%zu width=%zu reason=%u partition=%td "
                   "reduce=%td\n",
                   count, sizeof(T), unsigned(sum.reason()),
                   sum ? operations(*sum, graph::Operation::Partition) : -1,
                   sum ? operations(*sum, graph::Operation::Reduce) : -1);
      return 1;
    }
    auto result = sum->run(std::span<const T>{input});
    if (!result || *result != std::vector<T>{expected}) {
      std::fprintf(
          stderr,
          "sum run count=%zu width=%zu reason=%u size=%zu actual=%llu "
          "expected=%llu\n",
          count, sizeof(T), unsigned(result.reason()),
          result ? result->size() : 0u,
          result && !result->empty() ? (unsigned long long)result->front() : 0u,
          (unsigned long long)expected);
      return 2;
    }
    std::fill(input.begin(), input.end(), T{8});
    result = sum->run(std::span<const T>{input});
    if (!result || *result != std::vector<T>{T{0}}) {
      std::fprintf(
          stderr,
          "sum none count=%zu width=%zu reason=%u size=%zu actual=%llu\n",
          count, sizeof(T), unsigned(result.reason()),
          result ? result->size() : 0u,
          result && !result->empty() ? (unsigned long long)result->front()
                                     : 0u);
      return 3;
    }
    std::fill(input.begin(), input.end(), T{1});
    result = sum->run(std::span<const T>{input});
    if (!result || *result != std::vector<T>{static_cast<T>(count)})
      return 4;
    if (count > 1u) {
      std::fill(input.begin(), input.end(), T{1});
      input.front() = std::numeric_limits<T>::max();
      const auto overflow_sum = sum->run(std::span<const T>{input});
      if (overflow_sum || overflow_sum.reason() != Reason::ReduceSumOverflow)
        return 14;
    }
  }

  const std::array<T, 7u> input{T{0}, T{9}, T{8}, T{3}, T{31}, T{16}, T{5}};
  std::vector<T> expected;
  T total{};
  for (const T value : input)
    if (selected(value)) {
      expected.push_back(value);
      total += value;
    }
  auto observed = on(target)
                      .template map<T>("filter-source", input.size(),
                                       [](auto value) { return value; })
                      .branch([](auto values) {
                        const auto filtered = values.filter(
                            [](auto value) { return (value & T{7}) != T{0}; });
                        return outputs(filtered, filtered.count(),
                                       filtered.reduce(Reduce::Sum));
                      })
                      .compile();
  if (!observed || operations(*observed, graph::Operation::Partition) != 1)
    return 5;
  const auto result = observed->run(input);
  if (!result || std::get<0>(*result) != expected ||
      std::get<1>(*result) != static_cast<T>(expected.size()) ||
      std::get<2>(*result) != total)
    return 6;

  auto counted = on(target)
                     .template map<T>("filter-source", input.size(),
                                      [](auto value) { return value; })
                     .filter([](auto value) { return (value & T{7}) != T{0}; })
                     .count()
                     .compile();
  if (!counted || operations(*counted, graph::Operation::Partition) != 0)
    return 7;
  const auto count_result = counted->run(input);
  if (!count_result ||
      *count_result != std::vector<T>{static_cast<T>(expected.size())})
    return 8;

  constexpr std::size_t capacity = 257u;
  auto bounded = on(target)
                     .template input<Bounded<T>>(capacity)
                     .filter([](auto value) { return (value & T{7}) != T{0}; })
                     .reduce(Reduce::Sum)
                     .compile();
  if (!bounded || operations(*bounded, graph::Operation::Partition) != 0)
    return 9;
  for (const std::size_t active : {0u, 1u, 256u, 257u}) {
    std::vector<T> values(capacity, std::numeric_limits<T>::max());
    T expected_sum{};
    for (std::size_t i = 0u; i < active; ++i) {
      values[i] = static_cast<T>(i * 17u);
      if (selected(values[i]))
        expected_sum += values[i];
    }
    const std::array<T, 1u> logical{static_cast<T>(active)};
    const auto answer = bounded->run(std::span<const T>{values}, logical);
    if (!answer || *answer != std::vector<T>{expected_sum})
      return 10;
  }
  const std::array<T, 1u> overflow{static_cast<T>(capacity + 1u)};
  const std::vector<T> poison(capacity, std::numeric_limits<T>::max());
  const auto rejected = bounded->run(std::span<const T>{poison}, overflow);
  if (rejected || rejected.reason() != Reason::WorksetOverflow)
    return 11;

  auto minimum = on(target)
                     .template map<T>("filter-source", input.size(),
                                      [](auto value) { return value; })
                     .filter([](auto value) { return (value & T{7}) != T{0}; })
                     .reduce(Reduce::Min)
                     .compile();
  if (!minimum || operations(*minimum, graph::Operation::Partition) != 1)
    return 12;
  const auto minimum_result = minimum->run(input);
  if (!minimum_result || *minimum_result != std::vector<T>{T{3}})
    return 13;
  return 0;
}

} // namespace

[[nodiscard]] int CheckFusionBoundaries(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const Target target = backend == Backend::Cpu
                            ? Target::cpu(2u)
                            : rund::node::test_contract::target_for(backend);
  if (!CheckFilterDescriptorDomain())
    return 115;
  const int narrow = CheckFilteredSum<std::uint32_t>(target);
  const int wide = CheckFilteredSum<std::uint64_t>(target);
  if (narrow != 0 || wide != 0) {
    std::fprintf(stderr, "filter sum backend=%u narrow=%d wide=%d\n",
                 static_cast<unsigned>(backend), narrow, wide);
    return 100 + (narrow != 0 ? narrow : wide);
  }
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

} // namespace rund_node_flow_contract
