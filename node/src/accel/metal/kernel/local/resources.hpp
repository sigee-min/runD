#pragma once

#include <vector>

struct MetalKernelEntry final {
  std::shared_ptr<void> resource{};
  std::shared_ptr<MetalViewLowering> view{};
  MetalKernelOps ops{};
  ResetSpan resets{};
  bool barrier_before{};
};

struct MetalReset final {
  MetalResidentBufferResult resident{};
  reset::Range range{};
};

struct MetalKernelResources final {
  MetalAdapter *adapter{};
  InlineIndexedStorage<MetalKernelEntry, kInlineBoundStepCapacity> entries{};
  std::vector<MetalReset> resets{};
  std::shared_ptr<void> reset_pipeline{};
  submission::State<MetalKernelResources> submission{};
  PreparedMemory memory{};
  std::uint64_t dispatch_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t traffic = 0u;
  KernelPreparationMode mode{KernelPreparationMode::Standalone};
  bool shared_scratch{};

  [[nodiscard]] bool reserve(const std::size_t step_count) {
    entries.resize(step_count);
    return entries.valid();
  }

  [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

  [[nodiscard]] MetalKernelEntry *entry(const std::size_t index) noexcept {
    return entries.get(index);
  }
};
