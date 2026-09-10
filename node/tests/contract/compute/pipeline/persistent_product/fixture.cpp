#include "fixture.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../target/selection.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/backing.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <mutex>
#include <span>
#include <utility>

namespace rund_node_test_persistent_product {
namespace {

[[nodiscard]] std::uint32_t input_value(const std::size_t index) noexcept {
  return static_cast<std::uint32_t>((index * 17u + 11u) % 1009u);
}

[[nodiscard]] std::uint32_t output_value(const std::size_t index) noexcept {
  return input_value(index) * 3u + 7u;
}

} // namespace

PublicationSnapshot SnapshotPublication(
    const std::shared_ptr<rund::compute::detail::PipelineState> &pipeline) {
  if (pipeline == nullptr || pipeline->publication == nullptr) {
    return {};
  }
  std::lock_guard lock{pipeline->publication->gate};
  return PublicationSnapshot{
      .generation = pipeline->publication->generation,
      .payload_epoch = pipeline->publication->payload_epoch,
  };
}

std::uint64_t BackingVersion(PersistentProductBacking &backing) noexcept {
  std::lock_guard lock{
      rund::compute::detail::VirtualBackingAccess::gate(backing)};
  return rund::compute::detail::VirtualBackingAccess::version(backing);
}

std::uint64_t BackingRecovery(PersistentProductBacking &backing) noexcept {
  std::lock_guard lock{
      rund::compute::detail::VirtualBackingAccess::gate(backing)};
  return rund::compute::detail::VirtualBackingAccess::recovery_bytes(backing);
}

bool PrepareProduct(const rund::compute::Backend backend,
                    const std::uint64_t coordinates, PreparedProduct &prepared,
                    bool &unavailable) {
  using namespace rund::compute;
  unavailable = false;
  const std::size_t pages =
      static_cast<std::size_t>(coordinates * ProductFrameCapacity - 1u);
  // Exercise a physically short final page on every actual product run. The
  // sealed pointwise fill completes its Host frame, while the inactive native
  // lanes and backing publication remain limited to these logical elements.
  const std::size_t elements = pages * ProductPageElements - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*opened)
          .map<std::uint32_t>("persistent-product-e2e", ProductPageElements,
                              [](auto value) { return value * 3u + 7u; })
          .compile();
  prepared.input = std::make_shared<PersistentProductBacking>(
      elements * sizeof(std::uint32_t));
  prepared.output = std::make_shared<PersistentProductBacking>(
      elements * sizeof(std::uint32_t));
  std::vector<std::uint32_t> seeded(elements);
  prepared.expected.resize(elements);
  for (std::size_t index = 0u; index < elements; ++index) {
    seeded[index] = input_value(index);
    prepared.expected[index] = output_value(index);
  }
  const bool seeded_ok = prepared.input->seed(std::as_bytes(std::span{seeded}));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.input);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.output);
  auto state =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!seeded_ok || !state || state.value() == nullptr) {
    return false;
  }
  prepared.state = std::move(state).value();
  const auto &stream = prepared.state->pipeline->residency->stream();
  return stream.page_count() == pages &&
         stream.frame_capacity() == ProductFrameCapacity &&
         stream.epoch_count() == coordinates;
}

bool ExactOutput(const PreparedProduct &prepared) noexcept {
  std::vector<std::uint32_t> observed(prepared.expected.size());
  return prepared.output->observe(
             std::as_writable_bytes(std::span{observed})) &&
         observed == prepared.expected;
}

} // namespace rund_node_test_persistent_product

#endif
