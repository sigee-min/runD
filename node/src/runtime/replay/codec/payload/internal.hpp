#pragma once

#include <cstdint>

namespace rund::node::replay_detail::artifact::payload_codec {

[[nodiscard]] bool next_sequence(std::uint64_t previous, bool starts_at_zero,
                                 std::uint64_t delta,
                                 std::uint64_t &result) noexcept;

} // namespace rund::node::replay_detail::artifact::payload_codec
