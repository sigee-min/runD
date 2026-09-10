#include "support.hpp"

#include <cstring>
#include <limits>

namespace rund_node_test_virtual::product::scan {

bool uses_device_vsm(const rund::compute::Backend backend,
                     const rund::compute::ResidencyStats &stats) noexcept {
  return backend != rund::compute::Backend::Cpu &&
         stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
         stats.window_queue_call_count == 1u;
}

ScanGenerations scan_generations(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState>
        &state) noexcept {
  ScanGenerations result{};
  result.fill(std::numeric_limits<std::uint64_t>::max());
  if (state == nullptr) {
    return result;
  }
  const std::array<std::shared_ptr<rund::compute::detail::PipelineState>, 2u>
      pipelines{state->pipeline, state->alternate_pipeline};
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    const auto &pipeline = pipelines[index];
    if (pipeline == nullptr || pipeline->publication == nullptr) {
      return result;
    }
    std::lock_guard pipeline_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    result[index] = pipeline->publication->generation;
  }
  return result;
}

ScanControls scan_controls(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState>
        &state) noexcept {
  ScanControls result{};
  for (ScanControlIdentity &identity : result) {
    identity.generation = std::numeric_limits<std::uint64_t>::max();
    identity.poisoned = true;
  }
  if (state == nullptr) {
    return result;
  }
  const std::array<std::shared_ptr<rund::compute::detail::PipelineState>, 2u>
      pipelines{state->pipeline, state->alternate_pipeline};
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    const auto &pipeline = pipelines[index];
    if (pipeline == nullptr) {
      return result;
    }
    std::lock_guard pipeline_lock{pipeline->gate};
    result[index] = ScanControlIdentity{pipeline->native_generation,
                                        pipeline->native_parity,
                                        pipeline->control_poisoned};
  }
  return result;
}

bool scan_matches(MemoryVirtualBacking &backing,
                  const std::span<const std::uint32_t> input,
                  const bool inclusive) noexcept {
  std::vector<std::byte> bytes(input.size() * sizeof(std::uint32_t));
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

PersistentScanBacking::PersistentScanBacking(const std::size_t bytes)
    : bytes_(bytes) {}

std::uint64_t PersistentScanBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::VirtualBackingTier
PersistentScanBacking::tier() const noexcept {
  return rund::compute::VirtualBackingTier::Persistent;
}

std::uint32_t PersistentScanBacking::max_parallel_reads() const noexcept {
  return 2u;
}

rund::compute::Status PersistentScanBacking::read(
    const std::uint64_t offset,
    const std::span<std::byte> output) noexcept {
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  read_bytes_.fetch_add(output.size(), std::memory_order_relaxed);
  return rund::compute::Status::success();
}

rund::compute::Status PersistentScanBacking::write(
    const std::uint64_t offset,
    const std::span<const std::byte> input) noexcept {
  if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  std::memcpy(bytes_.data() + offset, input.data(), input.size());
  return rund::compute::Status::success();
}

bool PersistentScanBacking::seed(
    const std::span<const std::byte> input) noexcept {
  if (input.size() != bytes_.size()) {
    return false;
  }
  std::memcpy(bytes_.data(), input.data(), input.size());
  return true;
}

std::uint64_t PersistentScanBacking::read_bytes() const noexcept {
  return read_bytes_.load(std::memory_order_relaxed);
}

int InitializeScanFixture(ScanFixture &fixture,
                          const rund::compute::Backend backend) {
  using namespace rund::compute;
  fixture.backend = backend;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  fixture.device.emplace(std::move(device).value());
  for (std::size_t index = 0u; index < fixture.values.size(); ++index) {
    fixture.values[index] = static_cast<std::uint32_t>(index % 7u + 1u);
  }
  fixture.input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(fixture.values), ScanFrameElements * sizeof(std::uint32_t));
  if (!fixture.input_backing->seed(
          std::as_bytes(std::span{fixture.values}))) {
    return 3;
  }
  return 0;
}

int BeginTieredRoute(ScanFixture &fixture) {
  if (fixture.backend == rund::compute::Backend::Cpu) {
    return 0;
  }
  fixture.tiered_route = std::make_unique<DeviceVsmBypassScope>(*fixture.device);
  if (!fixture.tiered_route || !*fixture.tiered_route) {
    return 10;
  }
  return 0;
}

void ResetTieredRoute(ScanFixture &fixture) noexcept {
  fixture.tiered_route.reset();
}

} // namespace rund_node_test_virtual::product::scan
