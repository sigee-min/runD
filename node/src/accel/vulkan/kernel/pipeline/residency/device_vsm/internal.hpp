#pragma once

#include "../../../lease.hpp"
#include "../../../local.hpp"

#include "../../../../../clock.hpp"
#include "../../../../adapter/error.hpp"
#include "../../../../buffer/resident/lookup.hpp"
#include "../../../../buffer/resident/model.hpp"
#include "../../../../collective/pipeline.hpp"
#include "../../../../command/resources.hpp"
#include "../../../../descriptor.hpp"
#include "../../../../kernel.hpp"
#include "../../../../runtime/local.hpp"
#include "../../../../scope.hpp"
#include "graph_resident.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint64_t OwnerMagic = 0x56'44'56'53'4d'30'30'31ull;
inline constexpr char DescriptorCountOverflowReason[] =
    "device_vsm_descriptor_count_overflow";
inline constexpr std::size_t GraphBindingCapacity =
    3u * DeviceVsmResidentCapacity + DeviceVsmGraphResidentPhysicalCapacity +
    3u;

[[nodiscard]] inline bool descriptor_count_for(
    const bool graph, const bool window_ring, const bool ring,
    const std::uint64_t owner_count, const std::uint64_t endpoint_count,
    const std::uint64_t resident_count,
    std::uint32_t &descriptor_count) noexcept {
  std::uint64_t count = 0u;
  if (graph) {
    std::uint64_t owner_descriptors = 0u;
    if (!::rund::kernel::checked::mul(
            owner_count, DeviceVsmGraphResidentBankCapacity,
            owner_descriptors) ||
        !::rund::kernel::checked::add(3u, owner_descriptors, count) ||
        !::rund::kernel::checked::add(count, endpoint_count, count)) {
      return false;
    }
  } else if (window_ring) {
    count = 7u;
  } else if (ring) {
    if (!::rund::kernel::checked::mul(3u, resident_count, count) ||
        !::rund::kernel::checked::add(count, 3u, count)) {
      return false;
    }
  } else if (!::rund::kernel::checked::add(resident_count, 2u, count)) {
    return false;
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  descriptor_count = static_cast<std::uint32_t>(count);
  return true;
}

struct Dispatch final {
  std::uint32_t logical_elements{};
  std::uint32_t payload_elements{};
  std::uint32_t page_count{};
  std::uint32_t width{};
  std::uint32_t element_words{};
};

static_assert(sizeof(Dispatch) == 20u);

struct WindowRingDispatch final {
  std::uint32_t logical_elements{};
  std::uint32_t payload_elements{};
  std::uint32_t page_count{};
  std::uint32_t width{};
  std::uint32_t element_words{};
  std::uint32_t epoch{};
  std::uint32_t slot{};
  std::uint32_t phase{};
  std::uint32_t frame_elements{};
  std::uint32_t halo_elements{};
};

static_assert(sizeof(WindowRingDispatch) == 40u);

struct Owner final {
  std::mutex gate{};
  std::uint64_t magic{OwnerMagic};
  VulkanAdapter *adapter{};
  std::shared_ptr<void> adapter_owner{};
  rund::AccelDevice device{};
  std::shared_ptr<const DeviceVsmProof> proof{};
  VulkanCollectivePipeline *pipeline{};
  std::array<VulkanResidentBufferResult, DeviceVsmResidentCapacity> residents{};
  std::uint32_t resident_count{};
  ScopedBuffer params{};
  ScopedBuffer graph_table{};
  ScopedBuffer result{};
  ScopedBuffer ring_state{};
  std::array<ScopedBuffer, DeviceVsmResidentCapacity> ring_scratch{};
  std::uint32_t ring_scratch_count{};
  GraphResidentState graph{};
  std::array<VulkanStorageBinding, GraphBindingCapacity> graph_bindings{};
  std::uint32_t graph_descriptor_count{};
  VkDescriptorSet graph_descriptors{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases{};
  VulkanCommand graph_secondary{};
  VkPipeline graph_pipeline{VK_NULL_HANDLE};
  VkPipelineLayout graph_layout{VK_NULL_HANDLE};
  std::uint64_t graph_digest{};
  bool graph_recorded{};
  DeviceVsmCapability capability{};
  DeviceVsmRequest pending_request{};
  DeviceVsmNativeExecution pending_native{};
  std::uint64_t submit_begin_ns{};
  bool submitted{};
  bool in_flight{};
};

using GraphResidentBuffers = std::array<
    std::array<VulkanResidentBufferResult, DeviceVsmGraphResidentBankCapacity>,
    DeviceVsmGraphResidentPhysicalCapacity>;

[[nodiscard]] inline bool
same_ref(const rund::kernel::ResidentBufferRef &left,
         const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

[[nodiscard]] inline bool
preflight_graph_bindings(const rund::AccelDevice &pick,
                         const DeviceVsmGraphResidentProof &proof,
                         const GraphResidentBuffers *const expected,
                         GraphResidentBuffers &fresh) noexcept {
  if (!device_vsm_graph_resident_type_valid(proof.type) ||
      proof.owner_binding_count == 0u ||
      proof.owner_binding_count > DeviceVsmGraphResidentPhysicalCapacity) {
    return false;
  }
  for (std::size_t slot = 0u; slot < proof.owner_binding_count; ++slot) {
    const auto &source = proof.owners[slot];
    if (source.valid == 0u ||
        source.bank_count != DeviceVsmGraphResidentBankCapacity) {
      return false;
    }
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      VulkanResidentBufferResult current = LookupVulkanResidentBuffer(
          pick, source.refs[bank], source.handles[bank]);
      if (!current.check.ok || current.device_buffer == nullptr ||
          current.storage == nullptr ||
          !device_vsm_graph_resident_same_object(current.handle,
                                                 source.handles[bank])) {
        return false;
      }
      if (expected != nullptr) {
        const auto &old = (*expected)[slot][bank];
        if (!same_ref(current.ref, old.ref) ||
            !device_vsm_graph_resident_same_object(current.handle,
                                                   old.handle) ||
            !device_vsm_graph_resident_same_object(current.storage,
                                                   old.storage) ||
            current.device_buffer != old.device_buffer) {
          return false;
        }
      }
      fresh[slot][bank] = std::move(current);
    }
  }
  return true;
}

struct OwnerDelete final {
  Owner *owner{};
  void operator()(Owner *) const noexcept;
};

[[nodiscard]] std::shared_ptr<Owner> make_owner() noexcept;
[[nodiscard]] std::shared_ptr<Owner>
owner_of(const std::shared_ptr<void> &) noexcept;
[[nodiscard]] DeviceVsmRearmResult
rearm(const std::shared_ptr<void> &,
      const std::shared_ptr<const DeviceVsmProof> &) noexcept;
[[nodiscard]] bool
complete_frame_geometry(const DeviceVsmPageGeometry &) noexcept;
[[nodiscard]] DeviceVsmNativeExecution execute(Owner &, KernelCompletion,
                                               void *) noexcept;
[[nodiscard]] rund::AccelCheck submit(const DeviceVsmRequest &) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
