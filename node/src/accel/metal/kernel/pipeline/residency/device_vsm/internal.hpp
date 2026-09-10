#pragma once

#include "../../../../../kernel/residency/device_vsm/graph_wavefront.hpp"
#include "../../../../buffer/owner.hpp"
#include "../../../../buffer/resident/find.hpp"
#include "../../../../kernel.hpp"
#include "../../../local.hpp"
#include "../../icb.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail::metal_device_vsm {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

inline constexpr std::uint64_t OwnerMagic = 0x4d'44'56'53'4d'30'30'31ull;

struct Config final {
  std::uint64_t logical_elements{};
  std::uint32_t payload_elements{};
  std::uint32_t page_count{};
  std::uint32_t width{};
  std::uint32_t element_words{};
};

static_assert(sizeof(Config) == 24u);

struct WindowRingConfig final {
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

static_assert(sizeof(WindowRingConfig) == 40u);

struct GraphResidentTable final {
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

static_assert(offsetof(GraphResidentTable, logical_elements) == 0u);
static_assert(offsetof(GraphResidentTable, payload_elements) == 4u);
static_assert(offsetof(GraphResidentTable, page_count) == 8u);
static_assert(offsetof(GraphResidentTable, element_words) == 12u);
static_assert(offsetof(GraphResidentTable, element_bytes) == 16u);
static_assert(offsetof(GraphResidentTable, valid) == 20u);
static_assert(offsetof(GraphResidentTable, owner_count) == 24u);
static_assert(offsetof(GraphResidentTable, controller) == 28u);
static_assert(offsetof(GraphResidentTable, owner_valid) == 180u);
static_assert(offsetof(GraphResidentTable, owner_generations) == 216u);
static_assert(sizeof(GraphResidentTable) == 360u);

struct GraphResidentState final {
  std::uint32_t owner_count{};
  std::uint32_t endpoint_count{};
  std::uint32_t ready{};
  std::array<
      std::array<MetalResidentBufferResult, DeviceVsmGraphResidentBankCapacity>,
      DeviceVsmGraphResidentPhysicalCapacity>
      owners{};
  std::array<MetalResidentBufferResult, DeviceVsmResidentCapacity> endpoints{};
  std::shared_ptr<void> table{};
};

[[nodiscard]] inline bool
same_ref(const rund::kernel::ResidentBufferRef &left,
         const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

struct Owner final {
  std::mutex gate{};
  std::uint64_t magic{OwnerMagic};
  MetalAdapter *adapter{};
  std::shared_ptr<void> adapter_owner{};
  rund::AccelDevice device{};
  std::shared_ptr<const DeviceVsmProof> proof{};
  std::shared_ptr<void> pipeline{};
  std::shared_ptr<void> graph_icb{};
  std::shared_ptr<void> graph_pipeline{};
  std::uint64_t graph_digest{};
  std::array<MetalResidentBufferResult, DeviceVsmResidentCapacity> residents{};
  std::uint32_t resident_count{};
  std::shared_ptr<void> params{};
  std::shared_ptr<void> config{};
  std::shared_ptr<void> result{};
  GraphResidentState graph{};
  std::shared_ptr<void> ring_state{};
  std::array<std::shared_ptr<void>, DeviceVsmResidentCapacity> ring_scratch{};
  std::uint32_t ring_scratch_count{};
  DeviceVsmCapability capability{};
  DeviceVsmRequest pending_request{};
  DeviceVsmNativeExecution pending_native{};
  std::uint64_t submit_begin_ns{};
  bool submitted{};
  bool in_flight{};
};

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
[[nodiscard]] DeviceVsmNativeExecution execute(Owner &, KernelCompletion,
                                               void *) noexcept;
[[nodiscard]] rund::AccelCheck submit(const DeviceVsmRequest &) noexcept;

#endif

} // namespace rund::node::accel::detail::metal_device_vsm
