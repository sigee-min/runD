#include "model.hpp"

#include "../../target/selection.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_backend_contract {

template <class T> [[nodiscard]] bool CheckMovementDomainMatrix() {
  constexpr std::array<std::uint32_t, 4> indices{3u, 2u, 1u, 0u};
  constexpr std::array<std::uint32_t, 4> flags{0u, 1u, 0u, 1u};
  PrimitiveEvidence gather{};
  PrimitiveEvidence scatter{};
  PrimitiveEvidence partition{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckTwoInputPrimitive<T>(
            backend, "backend-gather", indices,
            [](auto values, const std::size_t count) {
              return std::move(values).gather(count);
            },
            gather) ||
        !CheckTwoInputPrimitive<T>(
            backend, "backend-scatter", indices,
            [](auto values, const std::size_t count) {
              return std::move(values).scatter(count, {.count = 4u});
            },
            scatter)) {
      std::fprintf(stderr, "movement failed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
    const std::array<T, 4u> priority_values{
        DomainValue<T>(1), DomainValue<T>(2), DomainValue<T>(3),
        DomainValue<T>(4)};
    constexpr std::uint32_t output_count = 64u;
    constexpr std::array<std::uint32_t, 4u> duplicate_first{0u, 0u,
                                                            output_count, 2u};
    constexpr std::array<std::uint32_t, 4u> range_first{output_count, 0u, 0u,
                                                        2u};
    const auto priority_target =
        rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
    auto priority_program =
        priority_target
            .template map<T>("backend-scatter-priority", priority_values.size(),
                             [](auto value) { return Store<T>(value); })
            .scatter(duplicate_first.size(), {.count = output_count})
            .compile();
    if (!priority_program) {
      return false;
    }
    auto priority_job =
        priority_program->resident(priority_values, duplicate_first);
    if (!priority_job) {
      return false;
    }
    const rund::compute::Status duplicate = priority_job->run();
    if (duplicate ||
        duplicate.reason() != rund::compute::Reason::ScatterDuplicateIndex ||
        !priority_job->write(priority_values, range_first)) {
      return false;
    }
    const rund::compute::Status range = priority_job->run();
    if (range ||
        range.reason() != rund::compute::Reason::ScatterIndexOutOfRange) {
      std::fprintf(stderr,
                   "scatter priority failed backend=%u bytes=%zu reason=%u\n",
                   static_cast<unsigned>(backend), sizeof(T),
                   static_cast<unsigned>(range.reason()));
      return false;
    }
    const std::array<T, 4> values{DomainValue<T>(3), DomainValue<T>(1),
                                  DomainValue<T>(4), DomainValue<T>(2)};
    const auto target =
        rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target
            .template map<T>("backend-partition", values.size(),
                             [](auto value) { return Store<T>(value); })
            .partition(flags.size())
            .compile();
    if (!program) {
      std::fprintf(
          stderr, "partition compile failed backend=%u bytes=%zu %.*s\n",
          static_cast<unsigned>(backend), sizeof(T),
          static_cast<int>(program.error().size()), program.error().data());
      return false;
    }
    auto job = program->resident(std::span<const T>{values},
                                 std::span<const std::uint32_t>{flags});
    if (!job || !job->run()) {
      std::fprintf(stderr, "partition run failed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
    const auto result = job->read();
    if (!result) {
      return false;
    }
    std::vector<std::uint64_t> bits;
    for (const T value : *result) {
      bits.push_back(DomainBits(value));
    }
    const auto stats = job->stats();
    if (backend == rund::compute::Backend::Cpu) {
      partition = PrimitiveEvidence{.values = std::move(bits),
                                    .graph = stats.graph_hash,
                                    .output = stats.output_hash};
    } else if (bits != partition.values ||
               stats.graph_hash != partition.graph ||
               stats.output_hash != partition.output) {
      std::fprintf(stderr, "partition parity failed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckPartialWriteResetWidth() {
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target =
        backend == rund::compute::Backend::Cpu
            ? rund::compute::on(rund::compute::Target::cpu(1u))
            : rund::compute::on(
                  rund::node::test_contract::target_for(backend, 2u));
    auto program = target
                       .map<T>("backend-partial-reset", 1u,
                               [](auto value) { return value; })
                       .scatter(1u, {.count = 2u})
                       .compile();
    constexpr std::array<T, 1u> first_value{7u};
    constexpr std::array<std::uint32_t, 1u> first_index{0u};
    if (!program) {
      return false;
    }
    auto job = program->resident(first_value, first_index);
    if (!job || !job->run()) {
      return false;
    }
    const auto first = job->read();
    const auto first_stats = job->stats();
    constexpr std::array<T, 2u> first_expected{7u, 0u};
    if (!first ||
        !std::equal(first->begin(), first->end(), first_expected.begin(),
                    first_expected.end()) ||
        first_stats.reset_bytes != 2u * sizeof(T) ||
        first_stats.reset_commands != 1u) {
      return false;
    }
    constexpr std::array<T, 1u> second_value{9u};
    constexpr std::array<std::uint32_t, 1u> second_index{1u};
    if (!job->write(second_value, second_index) || !job->run()) {
      return false;
    }
    const auto second = job->read();
    const auto second_stats = job->stats();
    constexpr std::array<T, 2u> second_expected{0u, 9u};
    if (!second ||
        !std::equal(second->begin(), second->end(), second_expected.begin(),
                    second_expected.end()) ||
        second_stats.reset_bytes != 2u * sizeof(T) ||
        second_stats.reset_commands != 1u) {
      std::fprintf(
          stderr,
          "partial reset failed backend=%u width=%zu bytes=%llu "
          "commands=%llu\n",
          static_cast<unsigned>(backend), sizeof(T),
          static_cast<unsigned long long>(second_stats.reset_bytes),
          static_cast<unsigned long long>(second_stats.reset_commands));
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckPartialWriteReset() {
  return CheckPartialWriteResetWidth<std::uint32_t>() &&
         CheckPartialWriteResetWidth<std::uint64_t>();
}

template <class T>
[[nodiscard]] constexpr T MovementValue(const std::uint64_t bits) noexcept {
  using Raw = typename T::Raw;
  using Unsigned = std::make_unsigned_t<Raw>;
  return T::from_raw(std::bit_cast<Raw>(static_cast<Unsigned>(bits)));
}

template <class T>
[[nodiscard]] constexpr std::array<T, 4u> ExactMovementInput() noexcept {
  if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return {MovementValue<T>(0x12345678u), MovementValue<T>(0x80010002u),
            MovementValue<T>(0x0000abcdu), MovementValue<T>(0x7fedcba9u)};
  } else {
    return {MovementValue<T>(0x123456789abcdef0ull),
            MovementValue<T>(0x8001000200030004ull),
            MovementValue<T>(0x0000abcd12345678ull),
            MovementValue<T>(0x7fedcba987654321ull)};
  }
}

template <class Range, class T, std::size_t Size>
[[nodiscard]] bool SameRawBits(const Range &actual,
                               const std::array<T, Size> &expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    if (actual[index].raw() != expected[index].raw()) {
      return false;
    }
  }
  return true;
}

template <class T>
[[nodiscard]] bool
ExactMovementGraph(const rund::compute::graph::Info &info) noexcept {
  using namespace rund::compute;
  if (!info.fingerprint || info.inputs.size() != 3u ||
      info.outputs.size() != 3u) {
    return false;
  }
  const auto resource = [&](const std::uint32_t id) -> const graph::Resource * {
    return id == 0u || id > info.resources.size() ? nullptr
                                                  : &info.resources[id - 1u];
  };
  const auto fixed = [](const graph::Resource &value,
                        const graph::Visibility visibility,
                        const Rounding rounding, const Overflow overflow,
                        const Approximation approximation) {
    return value.type == graph::Value::Fixed &&
           value.integer_bits == T::integer_bits &&
           value.fraction_bits == T::fraction_bits &&
           value.element_bytes == sizeof(T) && value.elements == 4u &&
           value.visibility == visibility && value.rounding == rounding &&
           value.overflow == overflow && value.approximation == approximation;
  };
  const auto u32_input = [](const graph::Resource &value) {
    return value.type == graph::Value::U32 && value.integer_bits == 0u &&
           value.fraction_bits == 0u &&
           value.visibility == graph::Visibility::Input &&
           value.element_bytes == sizeof(std::uint32_t) && value.elements == 4u;
  };
  const graph::Resource *const input = resource(info.inputs[0u]);
  const graph::Resource *const indices = resource(info.inputs[1u]);
  const graph::Resource *const flags = resource(info.inputs[2u]);
  if (input == nullptr || indices == nullptr || flags == nullptr ||
      !fixed(*input, graph::Visibility::Input, Rounding::NearestEven,
             Overflow::Saturate, Approximation::Exact) ||
      !u32_input(*indices) || !u32_input(*flags)) {
    return false;
  }
  for (const std::uint32_t id : info.outputs) {
    const graph::Resource *const output = resource(id);
    if (output == nullptr ||
        !fixed(*output, graph::Visibility::Output, Rounding::Down,
               Overflow::Wrap, Approximation::Deterministic)) {
      return false;
    }
  }
  std::size_t maps = 0u;
  std::size_t gathers = 0u;
  std::size_t scatters = 0u;
  std::size_t partitions = 0u;
  for (const graph::Node &node : info.nodes) {
    maps += node.operation == graph::Operation::Map ? 1u : 0u;
    gathers += node.operation == graph::Operation::Gather ? 1u : 0u;
    scatters += node.operation == graph::Operation::Scatter ? 1u : 0u;
    partitions += node.operation == graph::Operation::Partition ? 1u : 0u;
  }
  return info.nodes.size() == 4u && maps == 1u && gathers == 1u &&
         scatters == 1u && partitions == 1u;
}

template <class T> [[nodiscard]] bool CheckExactMovementMatrix() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> indices{3u, 1u, 0u, 2u};
  constexpr std::array<std::uint32_t, 4u> flags{1u, 0u, 1u, 0u};
  const std::array<T, 4u> input = ExactMovementInput<T>();
  PrimitiveEvidence reference{};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target = on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target.template input<T>(input.size())
            .template zip_input<std::uint32_t>(indices.size())
            .template zip_input<std::uint32_t>(flags.size())
            .branch([](auto values, auto order, auto mask) {
              const auto stored =
                  values.map("backend-exact-format-policy", [](auto value) {
                    return quantize<T, Rounding::Down, Overflow::Wrap,
                                    Approximation::Deterministic>(value);
                  });
              return outputs(stored.gather(order), stored.scatter(order),
                             stored.partition(mask));
            })
            .compile();
    if (!program || !ExactMovementGraph<T>(program->graph())) {
      return false;
    }
    auto job = program->resident(input, indices, flags);
    if (!job || !job->run()) {
      return false;
    }
    auto output = job->read_all();
    const Stats stats = job->stats();
    const std::array<T, 4u> gathered{input[3u], input[1u], input[0u],
                                     input[2u]};
    const std::array<T, 4u> scattered{input[2u], input[1u], input[3u],
                                      input[0u]};
    const std::array<T, 4u> partitioned{input[1u], input[3u], input[0u],
                                        input[2u]};
    if (!output || !SameRawBits(std::get<0>(*output), gathered) ||
        !SameRawBits(std::get<1>(*output), scattered) ||
        !SameRawBits(std::get<2>(*output), partitioned) ||
        stats.graph_hash == 0u || stats.output_hash == 0u) {
      return false;
    }
    std::vector<std::uint64_t> bits;
    bits.reserve(12u);
    const auto append = [&](const auto &values) {
      for (const T value : values) {
        using Raw = typename T::Raw;
        using Unsigned = std::make_unsigned_t<Raw>;
        bits.push_back(
            static_cast<std::uint64_t>(static_cast<Unsigned>(value.raw())));
      }
    };
    append(std::get<0>(*output));
    append(std::get<1>(*output));
    append(std::get<2>(*output));
    if (backend == Backend::Cpu) {
      reference = {.values = std::move(bits),
                   .graph = stats.graph_hash,
                   .output = stats.output_hash};
    } else if (bits != reference.values ||
               stats.graph_hash != reference.graph ||
               stats.output_hash != reference.output) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckMovementDomains() {
  return CheckMovementDomainMatrix<std::int32_t>() &&
         CheckMovementDomainMatrix<std::uint32_t>() &&
         CheckMovementDomainMatrix<std::int64_t>() &&
         CheckMovementDomainMatrix<std::uint64_t>() &&
         CheckMovementDomainMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckMovementDomainMatrix<rund::compute::Fixed<1, 63>>();
}

[[nodiscard]] bool CheckExactMovement() {
  return CheckExactMovementMatrix<rund::compute::Fixed<16, 16>>() &&
         CheckExactMovementMatrix<rund::compute::Fixed<20, 44>>();
}

} // namespace rund_node_backend_contract
