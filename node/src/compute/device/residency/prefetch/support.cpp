#include "local.hpp"

#include <chrono>

namespace rund::compute::detail::residency::prefetch_detail {

std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool alias_request_matches(const AliasLease &alias,
                           const PrefetchRequest &request,
                           const std::uint64_t token) noexcept {
  return alias.frame_bytes() != 0u && alias.owner_token() == token &&
         alias.source_key() == request.alias_source_key &&
         alias.target_key() == request.alias_target_key &&
         alias.target_key() == request.key &&
         alias.source_region() == request.alias_source_region &&
         alias.target_region() == request.alias_target_region &&
         alias.target_frame() == request.physical_frame &&
         alias.source_frame() == request.alias_source_frame &&
         alias.frame_bytes() == request.alias_frame_bytes &&
         alias.generation() == request.alias_generation &&
         alias.nonce() == request.alias_nonce &&
         alias.target_offset() == request.reuse_target_offset &&
         alias.bytes() == request.reuse_bytes &&
         alias.source_offset() == request.reuse_source_offset;
}

} // namespace rund::compute::detail::residency::prefetch_detail
