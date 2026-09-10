#include "../plan.hpp"

#include "internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency::execution {

bool Plan::input_bytes(const std::uint64_t page,
                       std::uint64_t &bytes) const noexcept {
  bytes = 0u;
  std::uint64_t offset = 0u;
  if (identity_ == 0u || page >= request_.page_count ||
      !plan_internal::checked_mul(page, request_.input.payload_bytes, offset) ||
      offset >= request_.input.logical_bytes) {
    return false;
  }
  bytes = std::min(request_.input.payload_bytes,
                   request_.input.logical_bytes - offset);
  return bytes != 0u;
}

bool Plan::input_source(const std::uint64_t page,
                        FetchSource &source) const noexcept {
  source = {};
  if (identity_ == 0u || page >= request_.page_count) {
    return false;
  }
  CacheKey key{};
  std::uint64_t core = 0u;
  if (!project_graph_cache_key(
          request_.input.cache,
          PageKey{.resource = request_.input.cache.resource, .page = page},
          key) ||
      !plan_internal::checked_mul(page, request_.input.payload_bytes, core) ||
      core >= request_.input.logical_bytes) {
    return false;
  }
  const std::uint64_t prefix = std::min(core, request_.input.read_prefix_bytes);
  const std::uint64_t offset = core - prefix;
  std::uint64_t core_end = 0u;
  if (!plan_internal::checked_add(core, request_.input.payload_bytes,
                                  core_end)) {
    return false;
  }
  const std::uint64_t clipped_core_end =
      std::min(core_end, request_.input.logical_bytes);
  const std::uint64_t available_suffix =
      request_.input.logical_bytes - clipped_core_end;
  const std::uint64_t end =
      clipped_core_end +
      std::min(request_.input.read_suffix_bytes, available_suffix);
  const std::uint64_t target = request_.input.target_prefix_bytes - prefix;
  if (end <= offset || target > request_.input.cache.page_bytes ||
      end - offset > request_.input.cache.page_bytes - target) {
    return false;
  }
  source = FetchSource{.key = key,
                       .offset = offset,
                       .bytes = end - offset,
                       .target_offset = target,
                       .frame_bytes = request_.input.cache.page_bytes,
                       .fill = request_.input.fill,
                       .fill_element_bytes = request_.input.fill_element_bytes,
                       .fill_value = request_.input.fill_value};
  return source.bytes != 0u;
}

bool Plan::input_reuse(const std::uint64_t page,
                       FetchReuseSource &reuse) const noexcept {
  reuse = {};
  FetchSource prior{};
  FetchSource current{};
  std::uint64_t prior_end = 0u;
  std::uint64_t current_end = 0u;
  if (identity_ == 0u || page == 0u || page >= request_.page_count ||
      !input_source(page - 1u, prior) || !input_source(page, current) ||
      !plan_internal::checked_add(prior.offset, prior.bytes, prior_end) ||
      !plan_internal::checked_add(current.offset, current.bytes, current_end) ||
      current.offset < prior.offset || current.offset >= prior_end ||
      current_end <= current.offset) {
    return false;
  }
  const std::uint64_t reuse_end = std::min(prior_end, current_end);
  const std::uint64_t bytes = reuse_end - current.offset;
  std::uint64_t source_offset = 0u;
  const std::uint64_t target_offset = current.target_offset;
  const std::uint64_t read_offset = reuse_end;
  std::uint64_t read_target_offset = 0u;
  const std::uint64_t read_bytes = current_end - reuse_end;
  if (bytes == 0u ||
      !plan_internal::checked_add(prior.target_offset,
                                  current.offset - prior.offset,
                                  source_offset) ||
      !plan_internal::checked_add(target_offset, bytes, read_target_offset) ||
      source_offset > prior.frame_bytes ||
      bytes > prior.frame_bytes - source_offset ||
      target_offset > current.frame_bytes ||
      bytes > current.frame_bytes - target_offset ||
      read_target_offset > current.frame_bytes ||
      read_bytes > current.frame_bytes - read_target_offset) {
    return false;
  }
  reuse = FetchReuseSource{.key = prior.key,
                           .source_offset = source_offset,
                           .target_offset = target_offset,
                           .bytes = bytes,
                           .read_offset = read_offset,
                           .read_target_offset = read_target_offset,
                           .read_bytes = read_bytes};
  return true;
}

bool Plan::input_live_rows(const std::size_t bank,
                           std::uint64_t &rows) const noexcept {
  rows = 0u;
  if (identity_ == 0u || bank >= BankCapacity) {
    return false;
  }
  if (request_.input.retain_until == NeverUse) {
    rows = request_.frame_capacity;
    return true;
  }
  if (epoch_count_ <= bank) {
    return true;
  }
  const std::uint64_t turns =
      1u +
      (epoch_count_ - 1u - static_cast<std::uint64_t>(bank)) / BankCapacity;
  if (!plan_internal::checked_mul(turns, request_.frame_capacity, rows)) {
    return false;
  }
  const std::uint64_t tail = request_.page_count % request_.frame_capacity;
  if (tail != 0u && (epoch_count_ - 1u) % BankCapacity == bank) {
    rows -= request_.frame_capacity - tail;
  }
  return true;
}

bool Plan::input_sources_materializable() const noexcept {
  if (identity_ == 0u || request_.page_count == 0u) {
    return false;
  }
  FetchSource first{};
  FetchSource last{};
  return input_source(0u, first) && first.materializes_frame() &&
         input_source(request_.page_count - 1u, last) &&
         last.materializes_frame();
}

} // namespace rund::compute::detail::residency::execution
