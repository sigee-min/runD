#include "local.hpp"

#include "../golden.hpp"

#include "../../../../target/selection.hpp"

#include <cstring>

namespace rund_node_test_virtual::product::failure {

OffsetFailPersistentBacking::OffsetFailPersistentBacking(
    const std::span<const std::int32_t> values)
    : OffsetFailPersistentBacking(std::as_bytes(values),
                                  ElementPageBytes * 2u) {}

OffsetFailPersistentBacking::OffsetFailPersistentBacking(
    const std::span<const std::byte> bytes, const std::uint64_t failure_offset)
    : bytes_(bytes.begin(), bytes.end()), failure_offset_(failure_offset) {}

std::uint64_t OffsetFailPersistentBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::VirtualBackingTier
OffsetFailPersistentBacking::tier() const noexcept {
  return rund::compute::VirtualBackingTier::Persistent;
}

std::uint32_t
OffsetFailPersistentBacking::max_parallel_reads() const noexcept {
  return 2u;
}

rund::compute::Status OffsetFailPersistentBacking::read(
    const std::uint64_t offset, const std::span<std::byte> output) noexcept {
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(
        rund::compute::Reason::TransferInvalid);
  }
  if (offset == failure_offset_ &&
      fail_.exchange(false, std::memory_order_acq_rel)) {
    return rund::compute::Status::fail(rund::compute::Reason::BackendFailed);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status OffsetFailPersistentBacking::write(
    std::uint64_t, std::span<const std::byte>) noexcept {
  return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
}

int InitializeFailureFixture(FailureFixture &fixture,
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
          .map<std::int32_t>("virtual-product-backing-retry", PageElements,
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

} // namespace rund_node_test_virtual::product::failure
