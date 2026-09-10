#include "internal.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstring>
#include <limits>

namespace rund_node_test_virtual::product::graph_wavefront_host {
namespace {

[[nodiscard]] std::size_t rounded_capacity(const std::size_t logical,
                                           const std::size_t page) noexcept {
  if (page == 0u) {
    return 0u;
  }
  const std::size_t remainder = logical % page;
  const std::size_t padding = remainder == 0u ? page : page - remainder;
  return logical <= std::numeric_limits<std::size_t>::max() - padding
             ? logical + padding
             : 0u;
}

} // namespace

ReversedGraphBacking::ReversedGraphBacking(const std::size_t logical_bytes,
                                           const std::size_t page_bytes,
                                           std::shared_ptr<PairReadGate> gate,
                                           const bool slow)
    : bytes_(rounded_capacity(logical_bytes, page_bytes)),
      logical_bytes_(logical_bytes), page_bytes_(page_bytes),
      gate_(std::move(gate)), slow_(slow) {
  if (gate_ != nullptr) {
    std::lock_guard lock{gate_->gate};
    gate_->page_bytes = page_bytes_;
  }
}

std::uint64_t ReversedGraphBacking::size_bytes() const noexcept {
  return logical_bytes_;
}

rund::compute::Status
ReversedGraphBacking::read(const std::uint64_t offset,
                           const std::span<std::byte> output) noexcept {
  if (!contains(offset, output.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (gate_ != nullptr) {
    std::unique_lock lock{gate_->gate};
    if (slow_ && !blocked_once_ && offset == 0u) {
      blocked_once_ = true;
      gate_->slow_blocked = true;
      gate_->ready.notify_all();
      if (!gate_->ready.wait_for(lock, std::chrono::seconds{2},
                                 [this] { return gate_->fast_completed; })) {
        gate_->timed_out = true;
        gate_->ready.notify_all();
        return rund::compute::Status::fail(rund::compute::Reason::PipelineBusy);
      }
      gate_->slow_released = true;
      gate_->ready.notify_all();
    }
    if (slow_ && gate_->slow_blocked && !gate_->slow_released &&
        offset >= gate_->page_bytes * 2u) {
      gate_->premature_reuse = true;
    }
    if (!slow_ && offset == 0u) {
      gate_->fast_started = true;
      gate_->ready.notify_all();
    }
  }

  {
    std::lock_guard lock{data_gate_};
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    ++facts_.read_count;
    facts_.read_bytes += output.size();
  }

  if (gate_ != nullptr) {
    std::lock_guard lock{gate_->gate};
    if (!slow_ && offset == 0u) {
      gate_->fast_completed = true;
      gate_->ready.notify_all();
    }
    if (slow_ && offset == 0u && blocked_once_) {
      gate_->slow_completed = true;
      gate_->reversed = gate_->fast_completed;
      gate_->ready.notify_all();
    }
  }
  return rund::compute::Status::success();
}

rund::compute::Status
ReversedGraphBacking::write(const std::uint64_t offset,
                            const std::span<const std::byte> input) noexcept {
  if (!contains(offset, input.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  std::lock_guard lock{data_gate_};
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  ++facts_.write_count;
  facts_.write_bytes += input.size();
  return rund::compute::Status::success();
}

bool ReversedGraphBacking::seed(
    const std::span<const std::byte> input) noexcept {
  if (input.size() != logical_bytes_) {
    return false;
  }
  std::lock_guard lock{data_gate_};
  std::memcpy(bytes_.data(), input.data(), input.size());
  return true;
}

BackingFacts ReversedGraphBacking::facts() const noexcept {
  std::lock_guard lock{data_gate_};
  return facts_;
}

bool ReversedGraphBacking::reversed() const noexcept {
  std::lock_guard lock{gate_->gate};
  return gate_->reversed && !gate_->timed_out;
}

bool ReversedGraphBacking::premature_reuse() const noexcept {
  std::lock_guard lock{gate_->gate};
  return gate_->premature_reuse;
}

bool ReversedGraphBacking::both_callbacks_started_before_release()
    const noexcept {
  std::lock_guard lock{gate_->gate};
  return gate_->slow_blocked && gate_->fast_started && gate_->fast_completed;
}

bool ReversedGraphBacking::contains(const std::uint64_t offset,
                                    const std::size_t bytes) const noexcept {
  return offset <= logical_bytes_ &&
         bytes <= logical_bytes_ - static_cast<std::size_t>(offset);
}

Preparation prepare_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_program(device);
  if (!program) {
    return {.reason = 2};
  }
  if (!validate_program(*program)) {
    return {.reason = 3};
  }

  std::array<std::uint64_t, ElementCount> first_values{};
  std::array<std::uint64_t, ElementCount> second_values{};
  std::array<std::uint64_t, ElementCount> third_values{};
  std::array<std::uint64_t, ElementCount> expected{};
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    first_values[index] = (index * 29u + 11u) % 127u;
    second_values[index] = (index * 17u + 7u) % 131u;
    third_values[index] = (index * 13u + 5u) % 137u;
    expected[index] = first_values[index] * StageLeafCount + stage_offset(1u) +
                      second_values[index] * StageLeafCount +
                      stage_offset(StageLeafCount + 1u) +
                      third_values[index] * StageLeafCount +
                      stage_offset(2u * StageLeafCount + 1u);
  }
  auto first_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(first_values), PageBytes);
  auto gate = std::make_shared<PairReadGate>();
  auto second_backing = std::make_shared<ReversedGraphBacking>(
      sizeof(second_values), PageBytes, gate, true);
  auto third_backing = std::make_shared<ReversedGraphBacking>(
      sizeof(third_values), PageBytes, gate, false);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(sizeof(expected), PageBytes);
  if (!first_backing->seed(std::as_bytes(std::span{first_values})) ||
      !second_backing->seed(std::as_bytes(std::span{second_values})) ||
      !third_backing->seed(std::as_bytes(std::span{third_values}))) {
    return {.reason = 4};
  }

  auto first = virtual_buffer<std::uint64_t>(ElementCount, first_backing);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, second_backing);
  auto third = virtual_buffer<std::uint64_t>(ElementCount, third_backing);
  auto output = virtual_buffer<std::uint64_t>(ElementCount, output_backing);
  if (!first || !second || !third || !output) {
    return {.reason = 5};
  }
  auto prepared = virtual_pipeline(*program, *first, *second, *third, *output,
                                   ResidencyConfig{});
  if (!prepared) {
    return {.reason = 6};
  }
  Pipeline pipeline = std::move(prepared).value();
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  if (state == nullptr ||
      state->geometry.route != detail::VirtualRoute::GraphPointwise ||
      state->geometry.device_vsm_required ||
      state->device_vsm_product_cache != nullptr) {
    return {.reason = 6};
  }
  return {.value = std::make_unique<Case>(Case{
              .first_values = std::move(first_values),
              .second_values = std::move(second_values),
              .third_values = std::move(third_values),
              .expected = std::move(expected),
              .first_backing = std::move(first_backing),
              .second_backing = std::move(second_backing),
              .third_backing = std::move(third_backing),
              .output_backing = std::move(output_backing),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

} // namespace rund_node_test_virtual::product::graph_wavefront_host
