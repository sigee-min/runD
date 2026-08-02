#pragma once

#include <vector>

enum class MetalKernelTemplateKind : std::uint8_t {
  Program,
  MapRecurrence,
};

struct MetalKernelProgramStepTemplate final {
  MetalKernelOps ops{};
  // Immutable, kind-specific pipeline/source recipe owner. Route state may
  // borrow it but never places buffer identity or mutable execution state in
  // this Program-level slot.
  std::shared_ptr<const void> immutable{};
  PreparedBackendManifest manifest{};
};

struct MetalKernelProgramTemplate final {
  // Every Metal owner published through the type-erased common registry starts
  // with this discriminator. Collision callbacks inspect it before casting.
  MetalKernelTemplateKind kind{MetalKernelTemplateKind::Program};
  const BackendRun *signature{};
  BackendTemplateRouteDemand route_demand{};
  std::vector<MetalKernelProgramStepTemplate> steps{};
};

static_assert(std::is_standard_layout_v<MetalKernelProgramTemplate>);

[[nodiscard]] inline MetalKernelTemplateKind
MetalKernelTemplateKindOf(const void *const prepared) noexcept {
  // A standard-layout object and its first member are pointer-interconvertible.
  return prepared == nullptr
             ? MetalKernelTemplateKind::Program
             : *reinterpret_cast<const MetalKernelTemplateKind *>(prepared);
}

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
  std::shared_ptr<MetalKernelProgramTemplate> program{};
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
