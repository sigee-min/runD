#include "reuse.hpp"

#include <cstring>

namespace rund::compute::detail::sliding_product_detail {

bool prepare_fetch_backing_slice(
    const SlidingProductRun &state,
    const residency::execution::SlidingFetch &fetch, std::byte *const target,
    FetchBackingSlice &slice) noexcept {
  slice = {};
  const residency::execution::FetchSource source = fetch.source();
  if (state.run == nullptr || target == nullptr ||
      source.target_offset > source.frame_bytes ||
      source.bytes > source.frame_bytes - source.target_offset) {
    return false;
  }
  slice = FetchBackingSlice{.offset = source.offset,
                            .target_offset = source.target_offset,
                            .bytes = source.bytes};
  if (!fetch.reuses_frame()) {
    return fetch.backing_bytes() == source.bytes;
  }
  const residency::execution::FetchReuseSource reuse = fetch.reuse();
  std::byte *const prior =
      virtual_host_input_frame(*state.run, fetch.reuse_frame());
  if (!reuse || prior == nullptr || fetch.reuse_frame() == fetch.frame() ||
      reuse.target_offset != source.target_offset ||
      reuse.bytes > source.bytes || reuse.source_offset > source.frame_bytes ||
      reuse.bytes > source.frame_bytes - reuse.source_offset ||
      reuse.target_offset > source.frame_bytes ||
      reuse.bytes > source.frame_bytes - reuse.target_offset ||
      reuse.read_target_offset != reuse.target_offset + reuse.bytes ||
      reuse.bytes > std::numeric_limits<std::uint64_t>::max() - source.offset ||
      reuse.read_offset != source.offset + reuse.bytes ||
      reuse.read_bytes != source.bytes - reuse.bytes ||
      reuse.read_target_offset > source.frame_bytes ||
      reuse.read_bytes > source.frame_bytes - reuse.read_target_offset ||
      fetch.backing_bytes() != reuse.read_bytes) {
    return false;
  }
  std::memmove(target + reuse.target_offset, prior + reuse.source_offset,
               static_cast<std::size_t>(reuse.bytes));
  slice = FetchBackingSlice{.offset = reuse.read_offset,
                            .target_offset = reuse.read_target_offset,
                            .bytes = reuse.read_bytes};
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
