#include "internal.hpp"

#include "src/compute/virtual/backing.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_multi {
namespace {

struct PrepareSnapshot final {
  std::uint64_t graph_hi{};
  std::uint64_t graph_lo{};
  std::array<BackingFacts, 4u> backings{};
};

[[nodiscard]] bool same_facts(const BackingFacts &lhs,
                              const BackingFacts &rhs) noexcept {
  return lhs.read_count == rhs.read_count && lhs.read_bytes == rhs.read_bytes &&
         lhs.write_count == rhs.write_count &&
         lhs.write_bytes == rhs.write_bytes &&
         lhs.observation_count == rhs.observation_count &&
         lhs.observation_bytes == rhs.observation_bytes &&
         lhs.read_failure_count == rhs.read_failure_count &&
         lhs.write_failure_count == rhs.write_failure_count &&
         lhs.partial_write_bytes == rhs.partial_write_bytes;
}

[[nodiscard]] PrepareSnapshot
snapshot(const Program &program, const MemoryVirtualBacking &first,
         const MemoryVirtualBacking &second, const MemoryVirtualBacking &third,
         const MemoryVirtualBacking &output) noexcept {
  const auto fingerprint = program.fingerprint();
  return {.graph_hi = fingerprint.hi,
          .graph_lo = fingerprint.lo,
          .backings = {first.facts(), second.facts(), third.facts(),
                       output.facts()}};
}

[[nodiscard]] bool unchanged(const PrepareSnapshot &before,
                             const PrepareSnapshot &after) noexcept {
  if (before.graph_hi != after.graph_hi || before.graph_lo != after.graph_lo) {
    return false;
  }
  for (std::size_t index = 0u; index < before.backings.size(); ++index) {
    if (!same_facts(before.backings[index], after.backings[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr std::uint64_t
stage_value(const std::uint64_t value, const std::uint64_t first) noexcept {
  constexpr std::uint64_t LiteralSum =
      StageLeafCount * (StageLeafCount + 1u) / 2u;
  return value * StageLeafCount + LiteralSum + (first - 1u) * StageLeafCount;
}

[[nodiscard]] std::size_t mapped_second(const std::size_t index,
                                        const std::size_t page_count) noexcept {
  constexpr std::size_t FrameCount = 2u;
  const std::size_t page = index / FrameElements;
  const std::size_t local = index % FrameElements;
  const std::size_t base = page / FrameCount * FrameCount;
  const std::size_t count = std::min(FrameCount, page_count - base);
  const std::size_t source = base + count - 1u - (page - base);
  return source * FrameElements + local;
}

} // namespace

Preparation prepare_case(const rund::compute::Device &device,
                         const std::size_t page_count) {
  using namespace rund::compute;
  if (page_count == 0u ||
      page_count > std::numeric_limits<std::size_t>::max() / FrameElements) {
    return {.reason = 1};
  }
  auto program = build_program(device);
  if (!program) {
    std::fprintf(stderr, "Graph multi program build reason=%u\n",
                 static_cast<unsigned>(program.reason()));
    return {.reason = 2};
  }
  if (!validate_program(*program)) {
    return {.reason = 2};
  }
  const std::size_t element_count = page_count * FrameElements - TailElements;
  std::vector<std::uint64_t> first_values(element_count);
  std::vector<std::uint64_t> second_values(element_count);
  std::vector<std::uint64_t> third_values(element_count);
  std::vector<std::uint64_t> expected(element_count);
  for (std::size_t index = 0u; index < element_count; ++index) {
    first_values[index] = index * 17u + 5u;
    second_values[index] = index * 29u + 11u;
    third_values[index] = index * 43u + 13u;
  }
  for (std::size_t index = 0u; index < element_count; ++index) {
    const std::size_t second_index = mapped_second(index, page_count);
    expected[index] =
        stage_value(first_values[index], 1u) +
        stage_value(second_values[second_index], StageLeafCount + 1u) +
        stage_value(third_values[index], 2u * StageLeafCount + 1u);
  }
  const std::size_t logical_bytes = element_count * sizeof(std::uint64_t);
  const std::size_t page_bytes = FrameElements * sizeof(std::uint64_t);
  auto first_backing =
      std::make_shared<MemoryVirtualBacking>(logical_bytes, page_bytes);
  auto second_backing =
      std::make_shared<MemoryVirtualBacking>(logical_bytes, page_bytes);
  auto third_backing =
      std::make_shared<MemoryVirtualBacking>(logical_bytes, page_bytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(logical_bytes, page_bytes);
  if (!first_backing->seed(std::as_bytes(std::span{first_values})) ||
      !second_backing->seed(std::as_bytes(std::span{second_values})) ||
      !third_backing->seed(std::as_bytes(std::span{third_values}))) {
    return {.reason = 3};
  }
  auto first = virtual_buffer<std::uint64_t>(element_count, first_backing);
  auto second = virtual_buffer<std::uint64_t>(element_count, second_backing);
  auto third = virtual_buffer<std::uint64_t>(element_count, third_backing);
  auto output = virtual_buffer<std::uint64_t>(element_count, output_backing);
  const auto fingerprint = program->fingerprint();
  const std::array<GraphPageMapEntry, 4u> entries{
      GraphPageMapEntry{.input = 0u,
                        .target_local = 0u,
                        .source_local = 0u,
                        .origin = PageOrigin::Begin},
      GraphPageMapEntry{.input = 0u,
                        .target_local = 1u,
                        .source_local = 1u,
                        .origin = PageOrigin::Begin},
      GraphPageMapEntry{.input = 1u,
                        .target_local = 0u,
                        .source_local = 0u,
                        .origin = PageOrigin::End},
      GraphPageMapEntry{.input = 1u,
                        .target_local = 1u,
                        .source_local = 1u,
                        .origin = PageOrigin::End},
  };
  const GraphPageMap page_map{
      .graph_hi = fingerprint.hi,
      .graph_lo = fingerprint.lo,
      .entries = entries,
  };
  const auto rejects = [&](const GraphPageMap map) noexcept {
    return !virtual_pipeline(*program, *first, *second, *third, *output, map);
  };
  const std::array<GraphPageMapEntry, 2u> bad_tail_entries{
      GraphPageMapEntry{.input = 0u,
                        .target_local = 0u,
                        .source_local = 1u,
                        .origin = PageOrigin::Begin},
      GraphPageMapEntry{.input = 0u,
                        .target_local = 1u,
                        .source_local = 0u,
                        .origin = PageOrigin::Begin},
  };
  const GraphPageMap bad_tail{
      .graph_hi = fingerprint.hi,
      .graph_lo = fingerprint.lo,
      .entries = bad_tail_entries,
  };
  const GraphPageMap bad_graph{
      .graph_hi = fingerprint.hi ^ 1u,
      .graph_lo = fingerprint.lo,
      .entries = entries,
  };
  const auto rejects_without_mutation = [&](const GraphPageMap map) noexcept {
    const PrepareSnapshot before =
        snapshot(*program, *first_backing, *second_backing, *third_backing,
                 *output_backing);
    const bool rejected = rejects(map);
    const PrepareSnapshot after =
        snapshot(*program, *first_backing, *second_backing, *third_backing,
                 *output_backing);
    return rejected && unchanged(before, after);
  };
  if ((page_count > 2u && !rejects_without_mutation(bad_tail)) ||
      !rejects_without_mutation(bad_graph)) {
    return {.reason = 5};
  }
  auto pipeline = first && second && third && output
                      ? virtual_pipeline(*program, *first, *second, *third,
                                         *output, page_map)
                      : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!pipeline) {
    const auto location = pipeline.location();
    std::fprintf(stderr, "Graph multi pipeline prepare reason=%u native=%s\n",
                 static_cast<unsigned>(pipeline.reason()),
                 location.native_reason_key == nullptr
                     ? "none"
                     : location.native_reason_key);
    return {.reason = 4};
  }
  Pipeline product = std::move(pipeline).value();
  const std::shared_ptr<detail::VirtualPipelineState> state =
      detail::VirtualPipelineAccess::state(product);
  return {.value = std::make_unique<Case>(Case{
              .page_count = page_count,
              .first_values = std::move(first_values),
              .second_values = std::move(second_values),
              .third_values = std::move(third_values),
              .expected = std::move(expected),
              .first_backing = std::move(first_backing),
              .second_backing = std::move(second_backing),
              .third_backing = std::move(third_backing),
              .output_backing = std::move(output_backing),
              .pipeline = std::move(product),
              .state = std::move(state),
          })};
}

OutputObservation observe_output(Case &test_case) noexcept {
  std::vector<std::uint64_t> observed(test_case.expected.size());
  if (!test_case.output_backing->observe(
          std::as_writable_bytes(std::span{observed}))) {
    return {};
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != test_case.expected[index]) {
      return {.readable = true,
              .matches = false,
              .first_bad_index = index,
              .actual = observed[index],
              .expected = test_case.expected[index]};
    }
  }
  return {.readable = true, .matches = true};
}

} // namespace rund_node_test_virtual::product::graph_pointwise_multi
