#include "local.hpp"

#include "backing.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <utility>

namespace rund_node_test_virtual::product {
namespace {

constexpr std::size_t ScanFrameElements = 16u;
constexpr std::size_t ScanElements = 53u;

[[nodiscard]] std::array<std::uint32_t, ScanElements> scan_input() noexcept {
  std::array<std::uint32_t, ScanElements> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(index % 7u + 1u);
  }
  return values;
}

[[nodiscard]] bool scan_matches(MemoryVirtualBacking &backing,
                                const std::span<const std::uint32_t> input,
                                const bool inclusive) noexcept {
  std::array<std::byte, ScanElements * sizeof(std::uint32_t)> bytes{};
  if (!backing.observe(bytes)) {
    return false;
  }
  std::uint32_t prefix = 0u;
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if (inclusive) {
      prefix += input[index];
    }
    std::uint32_t observed = 0u;
    std::memcpy(&observed, bytes.data() + index * sizeof(observed),
                sizeof(observed));
    if (observed != prefix) {
      return false;
    }
    if (!inclusive) {
      prefix += input[index];
    }
  }
  return true;
}

} // namespace

int CheckProductScan(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto inclusive_flow = on(*device).input<std::uint32_t>(ScanFrameElements);
  auto inclusive_program =
      std::move(inclusive_flow)
          .branch([](auto values) { return values.scan(Scan::InclusiveSum); })
          .compile();
  auto exclusive_flow = on(*device).input<std::uint32_t>(ScanFrameElements);
  auto exclusive_program =
      std::move(exclusive_flow)
          .branch([](auto values) { return values.scan(Scan::ExclusiveSum); })
          .compile();
  if (!inclusive_program || !exclusive_program ||
      inclusive_program->graph().nodes.size() != 1u ||
      exclusive_program->graph().nodes.size() != 1u) {
    return 2;
  }
  const auto values = scan_input();
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(values), ScanFrameElements * sizeof(std::uint32_t));
  if (!input_backing->seed(std::as_bytes(std::span{values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::uint32_t>(ScanElements, input_backing);
  auto inclusive_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
  auto inclusive_output =
      virtual_buffer<std::uint32_t>(ScanElements, inclusive_backing);
  auto inclusive =
      input && inclusive_output
          ? virtual_pipeline(*inclusive_program, *input, *inclusive_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  const Status inclusive_status =
      inclusive ? inclusive->run() : Status::fail(inclusive.reason());
  const bool inclusive_matches =
      inclusive_status && scan_matches(*inclusive_backing, values, true);
  if (!inclusive || !inclusive_status || !inclusive_matches) {
    std::fprintf(stderr,
                 "virtual scan inclusive prepared=%u reason=%.*s match=%u\n",
                 static_cast<unsigned>(static_cast<bool>(inclusive)),
                 static_cast<int>(inclusive_status.error().size()),
                 inclusive_status.error().data(),
                 static_cast<unsigned>(inclusive_matches));
    return 4;
  }
  constexpr std::size_t InclusivePages =
      (ScanElements + ScanFrameElements - 1u) / ScanFrameElements;
  const Stats inclusive_stats = inclusive->stats();
  if (inclusive->plan().residency.frame_capacity != 1u ||
      inclusive_stats.pipeline.residency.epoch_count != InclusivePages ||
      inclusive_stats.pipeline.residency.page_in_count != InclusivePages ||
      inclusive_stats.pipeline.residency.page_out_count != InclusivePages ||
      inclusive_stats.pipeline.residency.backing_read_bytes != sizeof(values) ||
      inclusive_stats.pipeline.residency.backing_write_bytes !=
          sizeof(values)) {
    return 5;
  }

  auto exclusive_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(values), sizeof(values));
  auto exclusive_output =
      virtual_buffer<std::uint32_t>(ScanElements, exclusive_backing);
  auto exclusive =
      input && exclusive_output
          ? virtual_pipeline(*exclusive_program, *input, *exclusive_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!exclusive || !exclusive->run() ||
      !scan_matches(*exclusive_backing, values, false)) {
    return 6;
  }
  constexpr std::size_t ExclusivePayload = ScanFrameElements - 1u;
  constexpr std::size_t ExclusivePages =
      (ScanElements + ExclusivePayload - 1u) / ExclusivePayload;
  const Stats exclusive_stats = exclusive->stats();
  if (exclusive->plan().residency.page_count != ExclusivePages ||
      exclusive_stats.pipeline.residency.page_in_count != ExclusivePages ||
      exclusive_stats.pipeline.residency.page_out_count != ExclusivePages ||
      exclusive_stats.pipeline.residency.backing_write_bytes !=
          sizeof(values)) {
    return 7;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
