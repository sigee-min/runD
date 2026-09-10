#pragma once

#include "../checkpoint.hpp"

namespace rund::replay::checkpoint_detail {

[[nodiscard]] std::uint64_t begin_state_hash(std::uint64_t schema,
                                             std::uint64_t state_size) noexcept;
[[nodiscard]] std::uint64_t
hash_state(std::uint64_t schema, std::span<const std::byte> state) noexcept;
[[nodiscard]] std::uint64_t
hash_boundary(std::uint64_t segment_count, std::uint64_t record_hash,
              std::uint64_t input_count, std::uint64_t input_hash,
              std::uint64_t transcript_hash) noexcept;
[[nodiscard]] std::uint64_t hash_prefix(std::uint64_t previous_prefix_hash,
                                        std::uint64_t segment_count,
                                        std::uint64_t boundary_hash) noexcept;
[[nodiscard]] std::uint64_t hash_transcript(
    std::uint64_t previous_transcript_prefix_hash,
    std::uint64_t previous_input_position, std::uint64_t segment_input_count,
    std::uint64_t segment_input_hash, std::uint64_t segment_transcript_hash,
    std::uint64_t input_position) noexcept;
[[nodiscard]] std::uint64_t
hash_checkpoint(std::uint64_t segment_count, std::uint64_t input_position,
                std::uint64_t state_size, std::uint64_t state_hash,
                std::uint64_t boundary_hash, std::uint64_t prefix_hash,
                std::uint64_t transcript_prefix_hash) noexcept;

} // namespace rund::replay::checkpoint_detail
