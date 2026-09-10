#include "../projection.hpp"
#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

bool project_virtual_epoch(const VirtualRunProjection &run,
                           const std::uint64_t epoch,
                           VirtualEpochProjection &projection) noexcept {
  projection = {};
  residency::PageRun pages{};
  if (!run.active.stream.epoch(epoch, pages) || pages.page_count == 0u ||
      pages.page_count > run.frame_capacity) {
    return false;
  }
  projection.failed_page = pages.first_page;
  projection.page_count = pages.page_count;
  std::uint64_t active_input_bytes = 0u;
  std::uint64_t active_output_bytes = 0u;
  if (!kernel::checked::mul(pages.first_page, run.input_payload_bytes,
                            projection.input_offset) ||
      (!run.reduction() &&
       !kernel::checked::mul(pages.first_page, run.output_payload_bytes,
                             projection.output_offset)) ||
      !kernel::checked::mul(pages.page_count, run.input_payload_bytes,
                            active_input_bytes) ||
      !kernel::checked::mul(pages.page_count, run.output_payload_bytes,
                            active_output_bytes) ||
      projection.input_offset >= run.active.input_bytes ||
      (!run.reduction() &&
       projection.output_offset >= run.active.output_bytes) ||
      active_input_bytes > std::numeric_limits<std::size_t>::max() ||
      active_output_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  projection.logical_input_bytes = static_cast<std::size_t>(std::min(
      active_input_bytes, run.active.input_bytes - projection.input_offset));
  projection.logical_output_bytes = static_cast<std::size_t>(
      run.reduction()
          ? active_output_bytes
          : std::min(active_output_bytes,
                     run.active.output_bytes - projection.output_offset));
  return true;
}

bool project_virtual_input_page(
    const VirtualRunProjection &run, const std::uint64_t page,
    VirtualInputPageProjection &projection) noexcept {
  projection = {};
  if (run.input_frame_elements == 0u || run.input_page_bytes == 0u ||
      run.input_page_bytes % run.input_frame_elements != 0u ||
      run.input_payload_bytes == 0u) {
    return false;
  }
  const std::uint64_t element_bytes =
      run.input_page_bytes / run.input_frame_elements;
  const std::uint64_t payload_elements =
      run.input_payload_bytes / element_bytes;
  const std::uint64_t prefix_elements = run.input_prefix_bytes / element_bytes;
  const std::uint64_t active_elements = run.input_visible_bytes / element_bytes;
  std::uint64_t core_first = 0u;
  std::uint64_t core_end = 0u;
  if (payload_elements == 0u || prefix_elements > run.input_frame_elements ||
      payload_elements > run.input_frame_elements - prefix_elements ||
      !kernel::checked::mul(page, payload_elements, core_first) ||
      core_first >= active_elements ||
      !kernel::checked::add(core_first, payload_elements, core_end)) {
    return false;
  }
  const std::uint64_t suffix_elements =
      run.input_frame_elements - prefix_elements - payload_elements;
  if (run.scan()) {
    std::uint64_t logical_offset = 0u;
    std::uint64_t target_offset = 0u;
    std::uint64_t transfer_elements =
        std::min(payload_elements, active_elements - core_first);
    std::uint64_t transfer_bytes = 0u;
    if (!kernel::checked::mul(core_first, element_bytes, logical_offset) ||
        !kernel::checked::mul(prefix_elements, element_bytes, target_offset) ||
        !kernel::checked::mul(transfer_elements, element_bytes,
                              transfer_bytes) ||
        logical_offset > std::numeric_limits<std::size_t>::max() ||
        target_offset > std::numeric_limits<std::size_t>::max() ||
        transfer_bytes > std::numeric_limits<std::size_t>::max() ||
        target_offset + transfer_bytes > run.input_page_bytes) {
      return false;
    }
    projection = VirtualInputPageProjection{
        .logical_offset = logical_offset,
        .target_offset = static_cast<std::size_t>(target_offset),
        .transfer_bytes = static_cast<std::size_t>(transfer_bytes),
        .leading_fill_bytes = static_cast<std::size_t>(target_offset),
        .trailing_fill_offset =
            static_cast<std::size_t>(target_offset + transfer_bytes),
    };
    return true;
  }
  const std::uint64_t first =
      core_first > prefix_elements ? core_first - prefix_elements : 0u;
  std::uint64_t wanted_end = 0u;
  if (!kernel::checked::add(core_end, suffix_elements, wanted_end)) {
    return false;
  }
  const std::uint64_t end = std::min(wanted_end, active_elements);
  const std::uint64_t target_elements =
      core_first >= prefix_elements ? 0u : prefix_elements - core_first;
  const std::uint64_t transfer_elements = end - first;
  std::uint64_t logical_offset = 0u;
  std::uint64_t target_offset = 0u;
  std::uint64_t transfer_bytes = 0u;
  if (end <= first ||
      !kernel::checked::mul(first, element_bytes, logical_offset) ||
      !kernel::checked::mul(target_elements, element_bytes, target_offset) ||
      !kernel::checked::mul(transfer_elements, element_bytes, transfer_bytes) ||
      logical_offset > std::numeric_limits<std::size_t>::max() ||
      target_offset > std::numeric_limits<std::size_t>::max() ||
      transfer_bytes > std::numeric_limits<std::size_t>::max() ||
      target_offset + transfer_bytes > run.input_page_bytes) {
    return false;
  }
  projection = VirtualInputPageProjection{
      .logical_offset = logical_offset,
      .target_offset = static_cast<std::size_t>(target_offset),
      .transfer_bytes = static_cast<std::size_t>(transfer_bytes),
      .leading_fill_bytes = static_cast<std::size_t>(target_offset),
      .trailing_fill_offset =
          static_cast<std::size_t>(target_offset + transfer_bytes),
  };
  return true;
}

bool project_virtual_input_reuse(
    const VirtualInputPageProjection &prior,
    const VirtualInputPageProjection &current,
    VirtualInputReuseProjection &projection) noexcept {
  projection = {};
  std::uint64_t prior_end = 0u;
  std::uint64_t current_end = 0u;
  if (prior.transfer_bytes == 0u || current.transfer_bytes == 0u ||
      !kernel::checked::add(prior.logical_offset, prior.transfer_bytes,
                            prior_end) ||
      !kernel::checked::add(current.logical_offset, current.transfer_bytes,
                            current_end) ||
      current.logical_offset < prior.logical_offset ||
      current.logical_offset >= prior_end || current_end <= prior_end) {
    return false;
  }
  const std::uint64_t overlap = prior_end - current.logical_offset;
  const std::uint64_t source =
      current.logical_offset - prior.logical_offset + prior.target_offset;
  const std::uint64_t target = current.target_offset;
  const std::uint64_t read_target = target + overlap;
  const std::uint64_t read_bytes = current_end - prior_end;
  if (overlap > std::numeric_limits<std::size_t>::max() ||
      source > std::numeric_limits<std::size_t>::max() ||
      target > std::numeric_limits<std::size_t>::max() ||
      read_target > std::numeric_limits<std::size_t>::max() ||
      read_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  projection = VirtualInputReuseProjection{
      .source_offset = static_cast<std::size_t>(source),
      .target_offset = static_cast<std::size_t>(target),
      .bytes = static_cast<std::size_t>(overlap),
      .logical_offset = prior_end,
      .read_target_offset = static_cast<std::size_t>(read_target),
      .read_bytes = static_cast<std::size_t>(read_bytes),
  };
  return true;
}

bool project_virtual_output_dirty(const VirtualRunProjection &run,
                                  const std::uint64_t page,
                                  residency::DirtyExtent &projection) noexcept {
  projection = {};
  residency::DirtyRange page_dirty{};
  std::uint64_t logical_offset = 0u;
  if (!run.active.stream.dirty_extent(page, page_dirty) ||
      page_dirty.bytes == 0u ||
      !kernel::checked::mul(page, run.output_payload_bytes, logical_offset) ||
      logical_offset >= run.active.stream.dirty_bytes() ||
      page_dirty.bytes > run.active.stream.dirty_bytes() - logical_offset) {
    return false;
  }
  projection = residency::DirtyExtent{.offset = logical_offset,
                                      .bytes = page_dirty.bytes};
  return true;
}

residency::CacheKey virtual_cache_key(const VirtualRunProjection &run,
                                      const std::uint64_t backing,
                                      const std::uint64_t version,
                                      const std::uint64_t page) noexcept {
  VirtualInputPageProjection projection{};
  const bool projected = project_virtual_input_page(run, page, projection);
  // A complete materialized frame is independent of the invocation's active
  // prefix. Only a right-boundary frame contains zero/identity/clamp fill and
  // therefore carries the exact active extent in its cache identity.
  const std::uint64_t extent =
      projected && projection.trailing_fill_offset == run.input_page_bytes
          ? 0u
          : run.cache_extent;
  return residency::CacheKey{
      .backing = backing,
      .version = version,
      .extent = extent,
      .materialization_hi = run.input_identity_hi,
      .materialization_lo = run.input_identity_lo,
      .page = page,
  };
}

residency::CacheKey virtual_output_cache_key(
    const VirtualRunProjection &run, const std::uint64_t backing,
    const std::uint64_t version, const std::uint64_t page) noexcept {
  const bool boundary =
      page + 1u == run.active.stream.page_count() &&
      run.output_payload_bytes != 0u &&
      run.active.output_bytes % run.output_payload_bytes != 0u;
  return residency::CacheKey{
      .backing = backing,
      .version = version,
      .extent = boundary ? run.cache_extent : 0u,
      .materialization_hi = run.result_identity_hi,
      .materialization_lo = run.result_identity_lo,
      .page = page,
      .domain = run.reduction() ? residency::CacheDomain::Transient
                                : residency::CacheDomain::Backing,
  };
}

} // namespace rund::compute::detail
