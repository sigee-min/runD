#include "../../internal.hpp"

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::uint64_t request_issue_key(const RequestIssue issue,
                                const std::size_t slot,
                                const std::size_t local) noexcept {
  std::uint64_t key = static_cast<std::uint64_t>(issue);
  if (slot != std::size_t(-1)) {
    key |= (static_cast<std::uint64_t>(slot) + 1u) << 16u;
  }
  if (local != std::size_t(-1)) {
    key |= (static_cast<std::uint64_t>(local) + 1u) << 32u;
  }
  return key;
}

std::uint64_t valid_request_issue() noexcept {
  return request_issue_key(RequestIssue::Valid);
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
