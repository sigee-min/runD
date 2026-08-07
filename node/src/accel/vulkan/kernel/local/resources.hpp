#pragma once

enum class VulkanKernelTemplateKind : std::uint8_t {
  Program,
  MapRecurrence,
};

enum class VulkanKernelDescriptorDependencyKind : std::uint8_t {
  Map,
  Collective,
};

// One collision-safe native dependency group, formed from the actual cache
// object pointer after immutable template materialization. Groups retain the
// exact route demand used for their one bulk reservation; route preparation
// may only lease from the frozen owner.
struct VulkanKernelDescriptorDependency final {
  VulkanKernelDescriptorDependencyKind kind{
      VulkanKernelDescriptorDependencyKind::Collective};
  const void *identity{};
  std::uint32_t descriptor_count{};
  std::uint64_t sets_per_route{};
  std::uint64_t set_capacity{};
  std::uint32_t earliest_order{};
  std::uint32_t source_node{NoNode};
  std::shared_ptr<VulkanMapDescriptorArena> map_arena{};
};

struct VulkanKernelProgramStepTemplate final {
  VulkanKernelOps ops{};
  // Immutable, kind-specific pipeline/source recipe owner. Descriptor leases,
  // bound buffers, and mutable execution state remain route-owned.
  std::shared_ptr<const void> immutable{};
  PreparedBackendManifest manifest{};
};

struct VulkanKernelProgramTemplate final {
  // Every Vulkan owner published through the type-erased common registry
  // starts with this discriminator. Match callbacks inspect it before a
  // concrete cast, so a cheap variant-hash collision remains memory-safe.
  VulkanKernelTemplateKind kind{VulkanKernelTemplateKind::Program};
  const BackendRun *signature{};
  BackendTemplateRouteDemand route_demand{};
  // Reset descriptor capacity is frozen with the shared Program template.
  // Keep the exact per-route set demand in its collision-safe identity so a
  // route with a different reset topology cannot borrow an undersized pool.
  std::uint64_t reset_set_count{};
  std::uint64_t view_set_count{};
  std::vector<VulkanKernelProgramStepTemplate> steps{};
  std::vector<VulkanKernelDescriptorDependency> descriptor_dependencies{};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanCollectivePipeline *reset_pipeline{};
  VulkanCollectivePipeline *view_pipeline{};
#endif
};

static_assert(std::is_standard_layout_v<VulkanKernelProgramTemplate>);

[[nodiscard]] inline VulkanKernelTemplateKind
VulkanKernelTemplateKindOf(const void *const prepared) noexcept {
  // A standard-layout object and its first member are pointer-interconvertible.
  return prepared == nullptr
             ? VulkanKernelTemplateKind::Program
             : *reinterpret_cast<const VulkanKernelTemplateKind *>(prepared);
}

struct VulkanKernelEntry final {
  std::shared_ptr<void> resource{};
  std::shared_ptr<VulkanViewLowering> view{};
  VulkanKernelOps ops{};
  ResetSpan resets{};
  bool barrier_before{};
};

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanReset final {
  VulkanResidentBufferResult resident{};
  reset::Range range{};
  VkDeviceSize binding_offset{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
  bool shader{};
};
#endif

struct VulkanKernelResources final {
  std::shared_ptr<VulkanKernelProgramTemplate> program{};
  InlineIndexedStorage<VulkanKernelEntry, kInlineBoundStepCapacity> entries{};
  std::uint64_t dispatch_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t traffic = 0u;
  KernelPreparationMode mode{KernelPreparationMode::Standalone};
  bool shared_scratch{};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *adapter = nullptr;
  VulkanCommand command{};
  VulkanCollectivePipeline *reset_pipeline{};
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases{};
  std::vector<VulkanReset> resets{};
  submission::State<VulkanKernelResources> submission{};
#endif

  [[nodiscard]] bool reserve(const std::size_t step_count) {
    entries.resize(step_count);
    return entries.valid();
  }

  [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

  [[nodiscard]] VulkanKernelEntry *entry(const std::size_t index) noexcept {
    return entries.get(index);
  }

  [[nodiscard]] const VulkanKernelEntry *
  entry(const std::size_t index) const noexcept {
    return entries.get(index);
  }

  void release() {
    VulkanKernelEntry *const values = entries.data();
    if (values == nullptr) {
      return;
    }
    for (std::size_t index = 0u; index < entries.size(); ++index) {
      values[index].resource.reset();
      values[index].view.reset();
    }
  }
};
