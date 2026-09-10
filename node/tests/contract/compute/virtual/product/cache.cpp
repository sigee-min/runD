#include "local.hpp"

#include "backing.hpp"
#include "cache/discard.hpp"
#include "cache/layout.hpp"
#include "cache/persistent.hpp"
#include "cache/reuse.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

int CheckProductDeviceCache(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-device-cache", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }

  auto input_backing = std::make_shared<MemoryVirtualBacking>(ElementPageBytes,
                                                              ElementPageBytes);
  auto first_backing = std::make_shared<MemoryVirtualBacking>(ElementPageBytes,
                                                              ElementPageBytes);
  auto second_backing = std::make_shared<MemoryVirtualBacking>(
      ElementPageBytes, ElementPageBytes);
  std::array<std::int32_t, PageElements> seeded{};
  SeedInput(seeded);
  if (!input_backing->seed(std::as_bytes(std::span{seeded}))) {
    return 3;
  }
  auto input = virtual_buffer<std::int32_t>(PageElements, input_backing);
  auto first = virtual_buffer<std::int32_t>(PageElements, first_backing);
  auto second = virtual_buffer<std::int32_t>(PageElements, second_backing);
  const ResidencyConfig config{
      .device_resident_bytes = ElementPageBytes * 4u,
      .host_resident_bytes = ElementPageBytes * 4u,
  };
  auto first_pipeline =
      input && first
          ? virtual_pipeline(*program, *input, *first, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  auto second_pipeline =
      input && second
          ? virtual_pipeline(*program, *input, *second, config)
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  int result = cache::CheckReuse(first_pipeline, second_pipeline,
                                 *input_backing, *second_backing);
  if (result != 0) {
    return result;
  }
  result = cache::CheckDiscard(program, config, seeded, second_pipeline);
  if (result != 0) {
    return result;
  }
  result = cache::CheckLayout(opened, second_pipeline);
  if (result != 0) {
    return result;
  }
  return cache::CheckStagedLoop(backend, opened, program);
}

} // namespace rund_node_test_virtual::product
