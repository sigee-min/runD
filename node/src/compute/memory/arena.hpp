#pragma once

#include <cstdint>

namespace rund::compute::detail {

struct DeviceState;

namespace memory {

inline constexpr std::uint64_t Alignment = 256u;
inline constexpr std::uint64_t Chunk = 1ull << 30u;
inline constexpr std::uint64_t Word = sizeof(std::uint32_t);
inline constexpr std::uint64_t AlignmentWords = Alignment / Word;
inline constexpr std::uint64_t ChunkWords = Chunk / Word;

static_assert((Alignment & (Alignment - 1u)) == 0u);
static_assert((Chunk & (Chunk - 1u)) == 0u);
static_assert(Chunk % Alignment == 0u);
static_assert(Alignment % Word == 0u);
static_assert(Chunk % Word == 0u);

[[nodiscard]] std::uint64_t
arena_bytes(const DeviceState &device) noexcept;

} // namespace memory
} // namespace rund::compute::detail
