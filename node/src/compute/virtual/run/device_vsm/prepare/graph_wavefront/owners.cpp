#include "../../../../../buffer/state.hpp"
#include "internal.hpp"

#include "../../../../../../accel/context/capability.hpp"
#include "../../../../../../compute/device/residency/pool.hpp"
#include "../../../../../../compute/device/state.hpp"
#include "../../../../../fixed/format.hpp"

#include <limits>
#include <memory>

namespace rund::compute::detail::device_vsm_product_detail {

bool project_graph_resident_owners(GraphResidentDraft &draft) noexcept {
  namespace accel = node::accel::detail;
  static_assert(residency::Pool::BankCount ==
                accel::DeviceVsmGraphResidentBankCapacity);
  constexpr std::size_t bank_count = residency::Pool::BankCount;
  for (std::size_t index = 0u; index < draft.internal_count; ++index) {
    const residency::TiledGraphPhysicalClass &owner =
        *draft.internal_owners[index];
    const residency::PoolPhysicalOwner *const physical =
        draft.pool.graph_owner(owner.physical_id);
    const residency::FrameRole expected_role =
        residency::FrameRole::Intermediate;
    if (owner.physical_id == 0u || physical == nullptr ||
        owner.type != draft.root_type || owner.page_bytes == 0u ||
        owner.page_bytes % draft.result.type.element_bytes != 0u ||
        owner.page_bytes / draft.result.type.element_bytes >
            std::numeric_limits<std::uint32_t>::max() ||
        !rund::kernel::ComputeFixedFormatAbsent(
            rund::compute::detail::kernel_format(owner.format)) ||
        physical->view == 0u || owner.color != physical->color ||
        physical->physical_class.page_bytes == 0u ||
        physical->physical_class.format != owner.format ||
        physical->physical_class.type != owner.type ||
        physical->physical_class.role != expected_role) {
      remember_graph_resident_reason(GraphResidentResourceOwnerInvalid,
                                     draft.reason);
      return false;
    }
    const std::shared_ptr<residency::PhysicalArena> &arena = physical->arena;
    if (arena == nullptr || physical->frame_capacity == 0u ||
        physical->frame_capacity < draft.plan.frame_capacity() ||
        physical->frame_capacity > std::numeric_limits<std::uint32_t>::max() ||
        arena->registry != draft.pool.registry || arena->extent == 0u ||
        arena->view == 0u || arena->bank_bytes == 0u ||
        arena->frame_capacity == 0u || arena->physical_class.page_bytes == 0u ||
        arena->bank_bytes % arena->physical_class.page_bytes != 0u ||
        arena->bank_bytes / arena->physical_class.page_bytes !=
            arena->frame_capacity ||
        arena->bank_bytes % physical->physical_class.page_bytes != 0u ||
        arena->bank_bytes / physical->physical_class.page_bytes !=
            physical->frame_capacity) {
      remember_graph_resident_reason(GraphResidentOwnerArenaInvalid,
                                     draft.reason);
      return false;
    }
    // Frame counts have different units when a byte extent is reblocked.
    // Authenticate each count against its own page geometry above, rather
    // than comparing the allocation's original pages with the typed view's.
    auto &row = draft.result.owners[index];
    row.physical_id = owner.physical_id;
    row.color = owner.color;
    row.view = physical->view;
    row.physical_format =
        rund::compute::detail::kernel_format(physical->physical_class.format);
    row.physical_page_bytes = physical->physical_class.page_bytes;
    row.physical_type =
        static_cast<std::uint8_t>(physical->physical_class.type);
    row.physical_tier =
        static_cast<std::uint8_t>(physical->physical_class.tier);
    row.physical_role =
        static_cast<std::uint8_t>(physical->physical_class.role);
    row.bank_count = static_cast<std::uint8_t>(bank_count);
    for (std::size_t bank = 0u; bank < bank_count; ++bank) {
      if (physical->buffers[bank] == nullptr ||
          physical->bank_regions[bank].count != draft.plan.frame_capacity() ||
          physical->cache_regions[bank].count != physical->frame_capacity ||
          arena->buffers[bank] == nullptr ||
          !same_physical_buffer(*physical->buffers[bank],
                                *arena->buffers[bank]) ||
          arena->bank_regions[bank].count != arena->frame_capacity) {
        remember_graph_resident_reason(GraphResidentBankBindingInvalid,
                                       draft.reason);
        return false;
      }
      const AccelBufferState *const storage =
          accel_buffer(*physical->buffers[bank]);
      if (storage == nullptr || !storage->buffer ||
          storage->buffer.resident.id == 0u ||
          storage->buffer.handle == nullptr) {
        remember_graph_resident_reason(GraphResidentBankBindingInvalid,
                                       draft.reason);
        return false;
      }
      const std::shared_ptr<node::accel::detail::AccelBufferToken> token =
          node::accel::detail::LookupAccelBufferToken(storage->buffer.handle);
      if (token == nullptr || !token->backend_check.ok ||
          token->backend_handle == nullptr) {
        remember_graph_resident_reason(GraphResidentBankBindingInvalid,
                                       draft.reason);
        return false;
      }
      row.refs[bank] = storage->buffer.resident;
      row.handles[bank] = token->backend_handle;
      const residency::FrameRegion cache = physical->cache_regions[bank];
      const residency::FrameRegion region = physical->bank_regions[bank];
      row.cache_regions[bank] = {.tier = static_cast<std::uint8_t>(cache.tier),
                                 .role = static_cast<std::uint8_t>(cache.role),
                                 .first = cache.first,
                                 .count = cache.count};
      row.bank_regions[bank] = {.tier = static_cast<std::uint8_t>(region.tier),
                                .role = static_cast<std::uint8_t>(region.role),
                                .first = region.first,
                                .count = region.count};
      if (storage->buffer.resident.offset_bytes != 0u) {
        remember_graph_resident_reason(GraphResidentBankBindingInvalid,
                                       draft.reason);
        return false;
      }
      row.local_first[bank] = cache.first;
      row.buffer_generations[bank] = arena->extent;
      if (row.cache_regions[bank].count == 0u ||
          row.bank_regions[bank].count == 0u ||
          row.buffer_generations[bank] == 0u ||
          row.refs[bank].element_bytes != draft.result.type.element_bytes ||
          row.refs[bank].stride_bytes != draft.result.type.element_bytes ||
          row.refs[bank].count == 0u || row.refs[bank].bytes == 0u) {
        remember_graph_resident_reason(GraphResidentBankBindingInvalid,
                                       draft.reason);
        return false;
      }
    }
    row.valid = 1u;
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
