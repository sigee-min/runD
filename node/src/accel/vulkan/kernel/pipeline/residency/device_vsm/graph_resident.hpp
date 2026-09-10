#pragma once

#include "../../../../../kernel/residency/device_vsm/graph_resident.hpp"
#include "../../../../../kernel/residency/device_vsm/graph_wavefront.hpp"

#include "../../../../buffer/resident/model.hpp"
#include "../../../../collective/pipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct GraphResidentProofTable final {
  std::uint32_t logical_elements{};
  std::uint32_t payload_elements{};
  std::uint32_t page_count{};
  std::uint32_t element_words{};
  std::uint32_t element_bytes{};
  std::uint32_t valid{};
  std::uint32_t owner_count{};
  DeviceVsmGraphController controller{};
  std::array<std::uint32_t, DeviceVsmGraphResidentPhysicalCapacity>
      owner_valid{};
  std::array<std::uint64_t, DeviceVsmGraphResidentPhysicalCapacity *
                                DeviceVsmGraphResidentBankCapacity>
      owner_generations{};
};

static_assert(offsetof(GraphResidentProofTable, logical_elements) == 0u);
static_assert(offsetof(GraphResidentProofTable, payload_elements) == 4u);
static_assert(offsetof(GraphResidentProofTable, page_count) == 8u);
static_assert(offsetof(GraphResidentProofTable, element_words) == 12u);
static_assert(offsetof(GraphResidentProofTable, element_bytes) == 16u);
static_assert(offsetof(GraphResidentProofTable, valid) == 20u);
static_assert(offsetof(GraphResidentProofTable, owner_count) == 24u);
static_assert(offsetof(GraphResidentProofTable, controller) == 28u);
static_assert(offsetof(GraphResidentProofTable, owner_valid) == 180u);
static_assert(offsetof(GraphResidentProofTable, owner_generations) == 216u);
static_assert(sizeof(GraphResidentProofTable) == 360u);

struct GraphResidentOwner final {
  std::array<VulkanResidentBufferResult, DeviceVsmGraphResidentBankCapacity>
      buffers{};
  std::uint32_t slot{};
  std::uint8_t valid{};
};

struct GraphResidentState final {
  std::array<GraphResidentOwner, DeviceVsmGraphResidentPhysicalCapacity>
      owners{};
  std::array<VulkanResidentBufferResult, DeviceVsmResidentCapacity> endpoints{};
  std::uint32_t owner_count{};
  std::uint32_t endpoint_count{};
  std::uint8_t ready{};
};

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
