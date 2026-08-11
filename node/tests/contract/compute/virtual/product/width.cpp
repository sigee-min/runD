#include "local.hpp"

#include "backing.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

int CheckProductWidthProjection(const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t InputPageBytes = PageElements * sizeof(std::uint64_t);
  constexpr std::size_t OutputPageBytes = PageElements * sizeof(std::uint32_t);
  constexpr std::size_t InputBytes = LogicalElements * sizeof(std::uint64_t);
  constexpr std::size_t OutputBytes = LogicalElements * sizeof(std::uint32_t);

  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::uint64_t>("virtual-width-projection", PageElements,
                              [](auto value) { return mask(value == value); })
          .compile();
  if (!program) {
    return 2;
  }

  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(InputBytes, InputPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(OutputBytes, OutputPageBytes);
  std::array<std::uint64_t, LogicalElements> input_values{};
  for (std::size_t index = 0u; index < input_values.size(); ++index) {
    input_values[index] = index + 17u;
  }
  if (!input_backing->seed(std::as_bytes(std::span{input_values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::uint64_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::uint32_t>(LogicalElements, output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::uint32_t(std::uint64_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || !prepared->run()) {
    return 4;
  }

  std::array<std::uint32_t, LogicalElements> observed{};
  if (!output_backing->observe(std::as_writable_bytes(std::span{observed}))) {
    return 5;
  }
  for (const std::uint32_t value : observed) {
    if (value != 1u) {
      return 6;
    }
  }

  const PipelinePlan plan = prepared->plan();
  const Stats stats = prepared->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  return plan.residency.logical_bytes == InputBytes + OutputBytes &&
                 plan.residency.page_bytes ==
                     InputPageBytes + OutputPageBytes &&
                 plan.residency.page_count == PageCount &&
                 plan.residency.frame_capacity == FrameCapacity &&
                 plan.residency.resident_bytes ==
                     FrameCapacity * 2u * (InputPageBytes + OutputPageBytes) &&
                 residency.active_count == LogicalElements &&
                 residency.backing_read_bytes == InputBytes &&
                 residency.backing_write_bytes == OutputBytes &&
                 stats.uploaded_bytes == PageCount * InputPageBytes &&
                 stats.downloaded_bytes == PageCount * OutputPageBytes
             ? 0
             : 7;
}

} // namespace rund_node_test_virtual::product
