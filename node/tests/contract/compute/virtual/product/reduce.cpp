#include "local.hpp"

#include "backing.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

constexpr std::size_t ReduceFrameElements = 16u;
constexpr std::size_t ReduceElements = 53u;
constexpr std::size_t ReducePages =
    (ReduceElements + ReduceFrameElements - 1u) / ReduceFrameElements;

template <class T>
[[nodiscard]] bool observed_scalar(MemoryVirtualBacking &backing,
                                   T &value) noexcept {
  std::array<std::byte, sizeof(T)> bytes{};
  if (!backing.observe(bytes)) {
    return false;
  }
  std::memcpy(&value, bytes.data(), sizeof(value));
  return true;
}

} // namespace

int CheckProductReduce(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto sum_program =
      on(*device)
          .map<std::uint64_t>("virtual-product-reduce-sum", ReduceFrameElements,
                              [](auto value) { return value; })
          .reduce(Reduce::Sum)
          .compile();
  if (!sum_program || sum_program->graph().nodes.empty() ||
      sum_program->graph().nodes.back().footprint.pattern !=
          graph::AccessPattern::Reduction) {
    return 2;
  }
  std::array<std::uint64_t, ReduceElements> values{};
  std::uint64_t expected_sum = 0u;
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = (index * 19u + 7u) % 101u;
    expected_sum += values[index];
  }
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(values), ReduceFrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  if (!input_backing->seed(std::as_bytes(std::span{values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::uint64_t>(ReduceElements, input_backing);
  auto output = virtual_buffer<std::uint64_t>(1u, output_backing);
  auto sum =
      input && output
          ? virtual_pipeline(*sum_program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                Reason::PipelineInvalid);
  std::uint64_t observed = 0u;
  if (!sum || !sum->run() || !observed_scalar(*output_backing, observed) ||
      observed != expected_sum) {
    return 4;
  }
  const Stats sum_stats = sum->stats();
  const ResidencyStats &sum_residency = sum_stats.pipeline.residency;
  if (sum->plan().residency.page_count != ReducePages ||
      sum_residency.page_in_count != ReducePages ||
      sum_residency.page_out_count != 1u ||
      sum_residency.backing_read_bytes != sizeof(values) ||
      sum_residency.backing_write_bytes != sizeof(std::uint64_t) ||
      sum_residency.page_out_bytes != sizeof(std::uint64_t) ||
      sum_stats.uploaded_bytes !=
          ReducePages * ReduceFrameElements * sizeof(std::uint64_t) ||
      sum_stats.downloaded_bytes != ReducePages * sizeof(std::uint64_t)) {
    return 5;
  }
  if (!sum->run(0u) || !observed_scalar(*output_backing, observed) ||
      observed != 0u || sum->stats().pipeline.residency.page_in_count != 0u ||
      sum->stats().pipeline.residency.page_out_count != 1u) {
    return 6;
  }

  auto min_program =
      on(*device)
          .map<std::int32_t>("virtual-product-reduce-min", ReduceFrameElements,
                             [](auto value) { return value; })
          .reduce(Reduce::Min)
          .compile();
  std::array<std::int32_t, ReduceElements> signed_values{};
  std::int32_t expected_min = std::numeric_limits<std::int32_t>::max();
  for (std::size_t index = 0u; index < signed_values.size(); ++index) {
    signed_values[index] = static_cast<std::int32_t>(index * 13u % 97u) - 45;
    expected_min = std::min(expected_min, signed_values[index]);
  }
  auto min_input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(signed_values), ReduceFrameElements * sizeof(std::int32_t));
  auto min_output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::int32_t), sizeof(std::int32_t));
  if (!min_program ||
      !min_input_backing->seed(std::as_bytes(std::span{signed_values}))) {
    return 7;
  }
  auto min_input =
      virtual_buffer<std::int32_t>(ReduceElements, min_input_backing);
  auto min_output = virtual_buffer<std::int32_t>(1u, min_output_backing);
  auto minimum =
      min_input && min_output
          ? virtual_pipeline(*min_program, *min_input, *min_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  std::int32_t observed_min = 0;
  if (!minimum || !minimum->run() ||
      !observed_scalar(*min_output_backing, observed_min) ||
      observed_min != expected_min ||
      minimum->run(0u).reason() != Reason::ReduceCountZero) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
