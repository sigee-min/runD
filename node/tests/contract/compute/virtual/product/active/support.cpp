#include "local.hpp"

#include <algorithm>
#include <cstring>

namespace rund_node_test_virtual::product::active {

[[nodiscard]] BackingFacts Delta(const BackingFacts after,
                                 const BackingFacts before) noexcept {
  return BackingFacts{
      .read_count = after.read_count - before.read_count,
      .read_bytes = after.read_bytes - before.read_bytes,
      .write_count = after.write_count - before.write_count,
      .write_bytes = after.write_bytes - before.write_bytes,
      .observation_count = after.observation_count - before.observation_count,
      .observation_bytes = after.observation_bytes - before.observation_bytes,
      .read_failure_count =
          after.read_failure_count - before.read_failure_count,
      .write_failure_count =
          after.write_failure_count - before.write_failure_count,
      .partial_write_bytes =
          after.partial_write_bytes - before.partial_write_bytes,
  };
}

[[nodiscard]] bool SameCapacity(
    const rund::compute::MemoryStats &left,
    const rund::compute::MemoryStats &right) noexcept {
  const auto same = [](const rund::compute::MemoryCounter a,
                       const rund::compute::MemoryCounter b) {
    return a.current == b.current;
  };
  return left.backend == right.backend && left.scope == right.scope &&
         same(left.host, right.host) && same(left.frame, right.frame) &&
         same(left.tile, right.tile) && same(left.resident, right.resident) &&
         same(left.staging, right.staging) && same(left.device, right.device) &&
         same(left.transfer, right.transfer);
}

[[nodiscard]] bool PrefixAndTail(const std::span<const std::byte> observed,
                                 const std::size_t active_count) noexcept {
  const std::size_t active_bytes = active_count * sizeof(std::int32_t);
  if (observed.size() != LogicalBytes || active_bytes > observed.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < active_count; ++index) {
    std::int32_t value = 0;
    std::memcpy(&value, observed.data() + index * sizeof(value), sizeof(value));
    if (value != ProductValue(SeedValue(index))) {
      return false;
    }
  }
  return std::all_of(observed.begin() +
                         static_cast<std::ptrdiff_t>(active_bytes),
                     observed.end(),
                     [](const std::byte value) { return value == TailPoison; });
}

[[nodiscard]] bool IsNativeMode(const RouteKind mode) noexcept {
  return mode == RouteKind::DeviceVsm || mode == RouteKind::Persistent;
}

int InitializeActiveFixture(ActiveFixture &fixture,
                            const rund::compute::Backend backend) {
  using namespace rund::compute;
  fixture.backend = backend;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  fixture.device.emplace(std::move(opened).value());
  auto program =
      on(*fixture.device)
          .map<std::int32_t>("virtual-product-active", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return 2;
  }
  fixture.input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  fixture.output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  SeedInput(fixture.seeded);
  for (std::size_t index = 0u; index < fixture.golden.size(); ++index) {
    fixture.golden[index] = ProductValue(fixture.seeded[index]);
  }
  if (!fixture.input_backing->seed(
          std::as_bytes(std::span{fixture.seeded}))) {
    return 3;
  }
  auto input =
      virtual_buffer<std::int32_t>(LogicalElements, fixture.input_backing);
  auto output =
      virtual_buffer<std::int32_t>(LogicalElements, fixture.output_backing);
  auto prepared =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 4;
  }
  fixture.program.emplace(std::move(program).value());
  fixture.input.emplace(std::move(input).value());
  fixture.output.emplace(std::move(output).value());
  fixture.prepared.emplace(std::move(prepared).value());
  return 0;
}

} // namespace rund_node_test_virtual::product::active
