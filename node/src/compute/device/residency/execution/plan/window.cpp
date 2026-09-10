#include "../plan.hpp"

#include "internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency::execution {

bool Plan::canonical_input_source(const std::uint64_t page,
                                  FetchSource &source) const noexcept {
  source = {};
  const GraphMaterialization &canonical = request_.canonical_input;
  if (identity_ == 0u || canonical.resource == 0u ||
      page >= canonical.page_count) {
    return false;
  }
  CacheKey key{};
  std::uint64_t offset = 0u;
  if (!project_graph_cache_key(
          canonical, PageKey{.resource = canonical.resource, .page = page},
          key) ||
      !plan_internal::checked_mul(page, canonical.page_bytes, offset) ||
      offset >= request_.input.logical_bytes) {
    return false;
  }
  const std::uint64_t bytes =
      std::min(canonical.page_bytes, request_.input.logical_bytes - offset);
  source = FetchSource{
      .key = key,
      .offset = offset,
      .bytes = bytes,
      .target_offset = 0u,
      .frame_bytes = canonical.page_bytes,
      .fill = bytes == canonical.page_bytes ? FetchFill::None
                                            : FetchFill::ZeroInactiveTail,
  };
  return source.materializes_frame();
}

bool Plan::window_footprint(
    const std::uint64_t epoch,
    WindowFootprintProjection &projection) const noexcept {
  projection = {};
  if (identity_ == 0u || request_.canonical_input.resource == 0u ||
      epoch >= epoch_count_ ||
      epoch >
          std::numeric_limits<std::uint64_t>::max() / request_.frame_capacity) {
    return false;
  }
  const std::uint64_t first_target = epoch * request_.frame_capacity;
  const std::size_t target_count = static_cast<std::size_t>(
      std::min(request_.page_count - first_target,
               static_cast<std::uint64_t>(request_.frame_capacity)));
  if (target_count == 0u || target_count > projection.targets.size()) {
    return false;
  }

  const auto source_range = [&](const std::uint64_t selected_epoch,
                                std::uint64_t &first,
                                std::uint64_t &last) noexcept {
    if (selected_epoch >= epoch_count_ ||
        selected_epoch > std::numeric_limits<std::uint64_t>::max() /
                             request_.frame_capacity) {
      return false;
    }
    const std::uint64_t target = selected_epoch * request_.frame_capacity;
    const std::uint64_t count =
        std::min(request_.page_count - target,
                 static_cast<std::uint64_t>(request_.frame_capacity));
    FetchSource leading{};
    FetchSource trailing{};
    std::uint64_t end = 0u;
    if (count == 0u || !input_source(target, leading) ||
        !input_source(target + count - 1u, trailing) ||
        !plan_internal::checked_add(trailing.offset, trailing.bytes, end) ||
        end == 0u) {
      return false;
    }
    first = leading.offset / request_.canonical_input.page_bytes;
    last = (end - 1u) / request_.canonical_input.page_bytes;
    return first <= last && last < request_.canonical_input.page_count;
  };

  std::uint64_t first_source = 0u;
  std::uint64_t last_source = 0u;
  std::uint64_t next_first = 0u;
  std::uint64_t next_last = 0u;
  const bool has_next = epoch + 1u < epoch_count_;
  if (!source_range(epoch, first_source, last_source) ||
      (has_next && !source_range(epoch + 1u, next_first, next_last)) ||
      last_source - first_source + 1u > projection.sources.size()) {
    return false;
  }

  projection.epoch = epoch;
  projection.target_count = target_count;
  for (std::size_t local = 0u; local < target_count; ++local) {
    const std::uint64_t page = first_target + local;
    FetchSource target{};
    std::uint64_t target_end = 0u;
    if (!input_source(page, target) ||
        !plan_internal::checked_add(target.offset, target.bytes, target_end)) {
      return false;
    }
    projection.targets[local] = target;
    std::uint64_t reconstructed = 0u;
    const std::uint64_t page_first =
        target.offset / request_.canonical_input.page_bytes;
    const std::uint64_t page_last =
        (target_end - 1u) / request_.canonical_input.page_bytes;
    for (std::uint64_t source_page = page_first; source_page <= page_last;
         ++source_page) {
      FetchSource canonical{};
      if (!canonical_input_source(source_page, canonical)) {
        return false;
      }
      std::size_t source_index = projection.source_count;
      for (std::size_t index = 0u; index < projection.source_count; ++index) {
        if (projection.sources[index].key == canonical.key) {
          source_index = index;
          break;
        }
      }
      if (source_index == projection.source_count) {
        if (projection.source_count == projection.sources.size()) {
          return false;
        }
        projection.sources[source_index] = CacheUse{
            .key = canonical.key,
            .access = Access::Read,
            .next_use = has_next && source_page >= next_first &&
                                source_page <= next_last
                            ? epoch + 1u
                            : NeverUse,
            .epoch = epoch,
            .retain_until = epoch,
        };
        ++projection.source_count;
      }
      std::uint64_t canonical_end = 0u;
      if (!plan_internal::checked_add(canonical.offset, canonical.bytes,
                                      canonical_end)) {
        return false;
      }
      const std::uint64_t begin = std::max(target.offset, canonical.offset);
      const std::uint64_t end = std::min(target_end, canonical_end);
      std::uint64_t target_offset = 0u;
      if (begin >= end ||
          !plan_internal::checked_add(target.target_offset,
                                      begin - target.offset, target_offset) ||
          target_offset > target.frame_bytes ||
          end - begin > target.frame_bytes - target_offset ||
          projection.slice_count == projection.slices.size() ||
          !plan_internal::checked_add(reconstructed, end - begin,
                                      reconstructed)) {
        return false;
      }
      projection.slices[projection.slice_count++] = WindowFootprintSlice{
          .source = static_cast<std::uint32_t>(source_index),
          .target = static_cast<std::uint32_t>(local),
          .source_offset = begin - canonical.offset,
          .target_offset = target_offset,
          .bytes = end - begin,
      };
    }
    if (reconstructed != target.bytes) {
      return false;
    }
  }
  return projection.source_count != 0u &&
         projection.source_count == last_source - first_source + 1u;
}

} // namespace rund::compute::detail::residency::execution
