#include "model.hpp"

#include <kernel/core/checked.hpp>

namespace rund::compute::resource::plan_detail {

Result<PhysicalAccess> physical_access(const Resource &resource,
                                       const Access &access) noexcept {
  if (resource.id == 0u || resource.alias_group == 0u ||
      !rund::kernel::checked::add(resource.alias_offset_bytes,
                                  resource.bytes)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceInvalid);
  }
  const bool contiguous = access.size_bytes != 0u;
  const bool strided = access.element_bytes != 0u ||
                       access.element_count != 0u || access.stride_bytes != 0u;
  if (contiguous == strided) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  const std::uint64_t element_bytes = contiguous ? 1u : access.element_bytes;
  const std::uint64_t element_count =
      contiguous ? access.size_bytes : access.element_count;
  const std::uint64_t stride_bytes = contiguous ? 1u : access.stride_bytes;
  std::uint64_t distance = 0u;
  std::uint64_t last = 0u;
  std::uint64_t relative_end = 0u;
  const bool footprint_overflow =
      element_count != 0u &&
      (stride_bytes == 0u ||
       !rund::kernel::checked::mul(element_count - 1u, stride_bytes,
                                   distance) ||
       !rund::kernel::checked::add(access.offset_bytes, distance, last) ||
       !rund::kernel::checked::add(last, element_bytes, relative_end));
  if (access.resource != resource.id || element_bytes == 0u ||
      element_bytes > 8u || element_count == 0u ||
      stride_bytes < element_bytes || footprint_overflow ||
      (access.mode != AccessMode::Read && access.mode != AccessMode::Write)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  if (relative_end > resource.bytes) {
    return Result<PhysicalAccess>::fail(Reason::ResourceAccessInvalid);
  }
  std::uint64_t physical_offset = 0u;
  std::uint64_t physical_end = 0u;
  if (!rund::kernel::checked::add(resource.alias_offset_bytes,
                                  access.offset_bytes, physical_offset) ||
      !rund::kernel::checked::add(resource.alias_offset_bytes, relative_end,
                                  physical_end)) {
    return Result<PhysicalAccess>::fail(Reason::ResourceRangeCapacity);
  }
  return Result<PhysicalAccess>::success(PhysicalAccess{
      .access = access,
      .alias_group = resource.alias_group,
      .offset = physical_offset,
      .element_bytes = element_bytes,
      .element_count = element_count,
      .stride_bytes = stride_bytes,
      .envelope_end = physical_end,
      .subtree_begin = physical_offset,
      .subtree_end = physical_end,
      .left = NoAccess,
      .right = NoAccess,
  });
}

} // namespace rund::compute::resource::plan_detail
