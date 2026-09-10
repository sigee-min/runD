#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include "src/compute/virtual/backing.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <span>
#include <utility>

namespace rund_node_test_persistent_product::window_test {
namespace {

[[nodiscard]] std::uint32_t input_value(const std::size_t index) noexcept {
  return static_cast<std::uint32_t>((index * 13u + 5u) % 101u);
}

[[nodiscard]] std::uint32_t output_value(const std::size_t index,
                                         const std::size_t count,
                                         const Edge edge) noexcept {
  std::uint32_t sum = 0u;
  for (std::size_t slot = 0u; slot < Radius * 2u + 1u; ++slot) {
    const std::size_t shifted = index + slot;
    if (shifted < Radius) {
      if (edge == Edge::Clamp) {
        sum += input_value(0u);
      }
      continue;
    }
    const std::size_t selected = shifted - Radius;
    if (selected < count) {
      sum += input_value(selected);
    } else if (edge == Edge::Clamp) {
      sum += input_value(count - 1u);
    }
  }
  return sum;
}

} // namespace

WindowBacking::WindowBacking(const std::size_t bytes) : bytes_(bytes) {}

std::uint64_t WindowBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status WindowBacking::read(
    const std::uint64_t offset,
    const std::span<std::byte> output) noexcept {
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  return rund::compute::Status::success();
}

rund::compute::Status WindowBacking::write(
    const std::uint64_t offset,
    const std::span<const std::byte> input) noexcept {
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  return rund::compute::Status::success();
}

bool WindowBacking::seed(const std::span<const std::byte> input) noexcept {
  if (input.size() != bytes_.size()) {
    return false;
  }
  std::memcpy(bytes_.data(), input.data(), input.size());
  return true;
}

bool WindowBacking::observe(const std::span<std::byte> output) const noexcept {
  if (output.size() != bytes_.size()) {
    return false;
  }
  std::memcpy(output.data(), bytes_.data(), output.size());
  return true;
}

std::uint64_t BackingVersion(WindowBacking &backing) noexcept {
  std::lock_guard lock{rund::compute::detail::VirtualBackingAccess::gate(
      backing)};
  return rund::compute::detail::VirtualBackingAccess::version(backing);
}

std::uint64_t BackingRecovery(WindowBacking &backing) noexcept {
  std::lock_guard lock{rund::compute::detail::VirtualBackingAccess::gate(
      backing)};
  return rund::compute::detail::VirtualBackingAccess::recovery_bytes(backing);
}

bool ExactWindowOutput(const WindowPrepared &prepared) noexcept {
  std::vector<std::uint32_t> observed(prepared.expected.size());
  return prepared.output != nullptr &&
         prepared.output->observe(std::as_writable_bytes(std::span{observed})) &&
         observed == prepared.expected;
}

bool PrepareWindowProduct(const rund::compute::Backend backend,
                          const std::uint64_t coordinates, const Edge edge,
                          WindowPrepared &prepared, bool &unavailable) {
  using namespace rund::compute;
  unavailable = false;
  const std::size_t pages = ProductPages(coordinates, edge);
  const std::size_t elements = ProductElements(coordinates, edge);
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  const WindowEdge window_edge =
      edge == Edge::Clamp ? WindowEdge::Clamp : WindowEdge::Clip;
  auto flow = on(*opened).input<std::uint32_t>(FrameElements);
  auto program = std::move(flow)
                     .branch([window_edge](auto values) {
                       return values.window(WindowSpec{
                           .op = Window::Sum,
                           .radius = Radius,
                           .edge = window_edge,
                       });
                     })
                     .compile();
  prepared.input =
      std::make_shared<WindowBacking>(elements * sizeof(uint32_t));
  prepared.output =
      std::make_shared<WindowBacking>(elements * sizeof(uint32_t));
  std::vector<std::uint32_t> seeded(elements);
  prepared.expected.resize(elements);
  for (std::size_t index = 0u; index < elements; ++index) {
    seeded[index] = input_value(index);
    prepared.expected[index] = output_value(index, elements, edge);
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
  return prepared.input->max_parallel_reads() == 1u &&
         prepared.output->max_parallel_reads() == 1u &&
         prepared.state->geometry.route == detail::VirtualRoute::Window &&
         stream.page_count() == pages && stream.frame_capacity() == 2u &&
         stream.epoch_count() == coordinates;
}

} // namespace rund_node_test_persistent_product::window_test

#endif
