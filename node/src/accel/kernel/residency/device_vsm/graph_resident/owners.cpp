#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::device_vsm_graph_resident {
namespace {

[[nodiscard]] bool region_valid(
    const std::uint64_t payload_elements,
    const DeviceVsmGraphResidentRegion &region,
    const std::uint32_t local_first,
    const rund::kernel::ResidentBufferRef &ref) noexcept {
  if (region.first < local_first) {
    return false;
  }
  const std::uint64_t relative =
      static_cast<std::uint64_t>(region.first - local_first);
  if (relative > std::numeric_limits<std::uint64_t>::max() - region.count) {
    return false;
  }
  const std::uint64_t frame_end = relative + region.count;
  if (frame_end >
          std::numeric_limits<std::uint64_t>::max() / payload_elements ||
      frame_end * payload_elements > ref.count) {
    return false;
  }
  return true;
}

} // namespace

bool validate_owners(const DeviceVsmGraphResidentProof &proof,
                     const ValidationContext &context) noexcept {
  for (std::size_t index = 0u; index < proof.owner_binding_count; ++index) {
    const auto &owner = proof.owners[index];
    if (owner.valid == 0u || owner.physical_id == 0u || owner.view == 0u ||
        owner.bank_count != DeviceVsmGraphResidentBankCapacity ||
        owner.physical_page_bytes == 0u ||
        owner.physical_page_bytes % proof.type.element_bytes != 0u ||
        owner.physical_page_bytes / proof.type.element_bytes >
            std::numeric_limits<std::uint32_t>::max() ||
        owner.physical_type != device_vsm_graph_resident_type_code(proof.type) ||
        !rund::kernel::ComputeFixedFormatAbsent(owner.physical_format)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (proof.owners[prior].physical_id == owner.physical_id &&
          proof.owners[prior].color == owner.color) {
        return false;
      }
    }
    for (std::size_t bank = 0u; bank < owner.bank_count; ++bank) {
      if (owner.refs[bank].id == 0u || owner.refs[bank].bytes == 0u ||
          owner.refs[bank].offset_bytes != 0u ||
          owner.refs[bank].bytes % proof.type.element_bytes != 0u ||
          owner.refs[bank].element_bytes != proof.type.element_bytes ||
          owner.refs[bank].stride_bytes != proof.type.element_bytes ||
          owner.refs[bank].count == 0u ||
          owner.refs[bank].count >
              owner.refs[bank].bytes / proof.type.element_bytes ||
          owner.handles[bank] == nullptr ||
          owner.cache_regions[bank].count == 0u ||
          owner.bank_regions[bank].count == 0u ||
          owner.buffer_generations[bank] == 0u ||
          owner.bank_regions[bank].first < owner.cache_regions[bank].first ||
          static_cast<std::uint64_t>(owner.bank_regions[bank].first -
                                     owner.cache_regions[bank].first) >
              std::numeric_limits<std::uint64_t>::max() -
                  owner.bank_regions[bank].count ||
          (static_cast<std::uint64_t>(owner.bank_regions[bank].first -
                                      owner.cache_regions[bank].first) +
               owner.bank_regions[bank].count >
           std::numeric_limits<std::uint64_t>::max() /
               context.payload_elements) ||
          (static_cast<std::uint64_t>(owner.bank_regions[bank].first -
                                      owner.cache_regions[bank].first) +
           owner.bank_regions[bank].count) * context.payload_elements >
              owner.refs[bank].count ||
          !region_valid(context.payload_elements, owner.cache_regions[bank],
                        owner.local_first[bank], owner.refs[bank]) ||
          !region_valid(context.payload_elements, owner.bank_regions[bank],
                        owner.local_first[bank], owner.refs[bank])) {
        return false;
      }
    }
  }
  return true;
}

bool validate_owner_usage(const DeviceVsmGraphResidentProof &proof,
                          const ValidationContext &context) noexcept {
  for (std::size_t owner = 0u; owner < proof.owner_binding_count; ++owner) {
    if (!context.owner_used[owner]) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident
