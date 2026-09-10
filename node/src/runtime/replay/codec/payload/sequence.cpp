#include "internal.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::replay_detail::artifact::payload_codec {

bool next_sequence(const std::uint64_t previous, const bool starts_at_zero,
                   const std::uint64_t delta, std::uint64_t &result) noexcept {
  std::uint64_t base = 0u;
  if (!starts_at_zero && !rund::kernel::checked::add(previous, 1u, base)) {
    return false;
  }
  return rund::kernel::checked::add(base, delta, result);
}

} // namespace rund::node::replay_detail::artifact::payload_codec
