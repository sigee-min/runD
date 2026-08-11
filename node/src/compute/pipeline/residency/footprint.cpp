#include "footprint.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <new>

namespace rund::compute::detail::residency {

bool ProjectRange(const ByteRange &range, const std::uint64_t page_bytes,
                  std::vector<PageUse> &uses) noexcept {
  if (range.resource == 0u || page_bytes == 0u || range.bytes == 0u) {
    return false;
  }
  std::uint64_t end = 0u;
  if (!kernel::checked::add(range.offset, range.bytes - 1u, end)) {
    return false;
  }
  const std::uint64_t first_page = range.offset / page_bytes;
  const std::uint64_t last_page = end / page_bytes;
  if (last_page - first_page == std::numeric_limits<std::uint64_t>::max() ||
      last_page - first_page + 1u >
          std::numeric_limits<std::size_t>::max() - uses.size()) {
    return false;
  }
  try {
    uses.reserve(uses.size() +
                 static_cast<std::size_t>(last_page - first_page + 1u));
    for (std::uint64_t page = first_page;; ++page) {
      uses.push_back(PageUse{
          .key = PageKey{.resource = range.resource, .page = page},
          .access = range.access,
      });
      if (page == last_page) {
        break;
      }
    }
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool ProjectWindow(const WindowFootprint &footprint,
                   const std::uint64_t page_bytes,
                   DemandEpoch &epoch) noexcept {
  if (footprint.input_resource == 0u || footprint.output_resource == 0u ||
      footprint.input_resource == footprint.output_resource ||
      footprint.input_elements == 0u || footprint.output_count == 0u ||
      footprint.window_size == 0u || footprint.stride == 0u ||
      footprint.element_bytes == 0u) {
    return false;
  }
  std::uint64_t output_last = 0u;
  std::uint64_t first_anchor = 0u;
  std::uint64_t last_anchor = 0u;
  if (!kernel::checked::add(footprint.output_first, footprint.output_count - 1u,
                            output_last) ||
      !kernel::checked::mul(footprint.output_first, footprint.stride,
                            first_anchor) ||
      !kernel::checked::mul(output_last, footprint.stride, last_anchor)) {
    return false;
  }
  const std::uint64_t first_input = first_anchor > footprint.pad_left
                                        ? first_anchor - footprint.pad_left
                                        : 0u;
  const std::uint64_t shifted_last =
      last_anchor > footprint.pad_left ? last_anchor - footprint.pad_left : 0u;
  std::uint64_t last_input = 0u;
  if (!kernel::checked::add(shifted_last, footprint.window_size - 1u,
                            last_input)) {
    return false;
  }
  last_input = std::min(last_input, footprint.input_elements - 1u);
  std::uint64_t input_offset = 0u;
  std::uint64_t input_bytes = 0u;
  std::uint64_t output_offset = 0u;
  std::uint64_t output_bytes = 0u;
  if (first_input > last_input ||
      !kernel::checked::mul(first_input, footprint.element_bytes,
                            input_offset) ||
      !kernel::checked::mul(last_input - first_input + 1u,
                            footprint.element_bytes, input_bytes) ||
      !kernel::checked::mul(footprint.output_first, footprint.element_bytes,
                            output_offset) ||
      !kernel::checked::mul(footprint.output_count, footprint.element_bytes,
                            output_bytes)) {
    return false;
  }
  epoch.uses.clear();
  return ProjectRange(ByteRange{.resource = footprint.input_resource,
                                .access = Access::Read,
                                .offset = input_offset,
                                .bytes = input_bytes},
                      page_bytes, epoch.uses) &&
         ProjectRange(ByteRange{.resource = footprint.output_resource,
                                .access = Access::Write,
                                .offset = output_offset,
                                .bytes = output_bytes},
                      page_bytes, epoch.uses);
}

bool ProjectScan(const ScanFootprint &footprint, const std::uint64_t page_bytes,
                 std::vector<DemandEpoch> &epochs) noexcept {
  if (footprint.input_resource == 0u || footprint.output_resource == 0u ||
      footprint.partial_resource == 0u ||
      footprint.input_resource == footprint.output_resource ||
      footprint.element_count == 0u || footprint.tile_elements == 0u ||
      footprint.element_bytes == 0u) {
    return false;
  }
  const std::uint64_t tile_count =
      footprint.element_count / footprint.tile_elements +
      static_cast<std::uint64_t>(
          footprint.element_count % footprint.tile_elements != 0u);
  if (tile_count > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  try {
    epochs.clear();
    epochs.reserve(static_cast<std::size_t>(tile_count * 2u + 1u));
    for (std::uint64_t tile = 0u; tile < tile_count; ++tile) {
      const std::uint64_t first = tile * footprint.tile_elements;
      const std::uint64_t count =
          std::min(footprint.tile_elements, footprint.element_count - first);
      std::uint64_t offset = 0u;
      std::uint64_t bytes = 0u;
      if (!kernel::checked::mul(first, footprint.element_bytes, offset) ||
          !kernel::checked::mul(count, footprint.element_bytes, bytes)) {
        return false;
      }
      DemandEpoch local{.node = 0u, .tile = static_cast<std::uint32_t>(tile)};
      if (!ProjectRange(ByteRange{.resource = footprint.input_resource,
                                  .access = Access::Read,
                                  .offset = offset,
                                  .bytes = bytes},
                        page_bytes, local.uses) ||
          !ProjectRange(ByteRange{.resource = footprint.output_resource,
                                  .access = Access::Write,
                                  .offset = offset,
                                  .bytes = bytes},
                        page_bytes, local.uses) ||
          !ProjectRange(ByteRange{.resource = footprint.partial_resource,
                                  .access = Access::Write,
                                  .offset = tile * footprint.element_bytes,
                                  .bytes = footprint.element_bytes},
                        page_bytes, local.uses)) {
        return false;
      }
      epochs.push_back(std::move(local));
    }
    DemandEpoch prefix{.node = 1u};
    if (!ProjectRange(ByteRange{.resource = footprint.partial_resource,
                                .access = Access::ReadWrite,
                                .bytes = tile_count * footprint.element_bytes},
                      page_bytes, prefix.uses)) {
      return false;
    }
    epochs.push_back(std::move(prefix));
    for (std::uint64_t tile = 0u; tile < tile_count; ++tile) {
      const std::uint64_t first = tile * footprint.tile_elements;
      const std::uint64_t count =
          std::min(footprint.tile_elements, footprint.element_count - first);
      DemandEpoch uniform{.node = 2u, .tile = static_cast<std::uint32_t>(tile)};
      if (!ProjectRange(ByteRange{.resource = footprint.output_resource,
                                  .access = Access::ReadWrite,
                                  .offset = first * footprint.element_bytes,
                                  .bytes = count * footprint.element_bytes},
                        page_bytes, uniform.uses) ||
          !ProjectRange(ByteRange{.resource = footprint.partial_resource,
                                  .access = Access::Read,
                                  .offset = tile * footprint.element_bytes,
                                  .bytes = footprint.element_bytes},
                        page_bytes, uniform.uses)) {
        return false;
      }
      epochs.push_back(std::move(uniform));
    }
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

} // namespace rund::compute::detail::residency
