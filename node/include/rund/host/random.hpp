#pragma once

#include <rund/host/random/seed.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::host::random {

struct StreamId {
  std::uint64_t value = 0u;
};

struct DrawId {
  std::uint64_t value = 0u;
};

struct Stream {
  std::uint64_t seed = 0u;
  std::uint64_t id = 0u;
};

[[nodiscard]] Stream stream(RunSeed seed, StreamId id) noexcept;
[[nodiscard]] Stream stream(StreamId id) noexcept;
[[nodiscard]] RunSeed active_run_seed() noexcept;
[[nodiscard]] Stream split(Stream parent, StreamId child) noexcept;
[[nodiscard]] std::uint64_t u64(Stream stream, DrawId draw) noexcept;
[[nodiscard]] std::uint32_t u32(Stream stream, DrawId draw) noexcept;
[[nodiscard]] std::uint64_t below(Stream stream, DrawId draw,
                                  std::uint64_t upper_exclusive) noexcept;
[[nodiscard]] std::uint64_t range(Stream stream, DrawId draw,
                                  std::uint64_t lower_inclusive,
                                  std::uint64_t upper_exclusive) noexcept;
[[nodiscard]] float unit_f32(Stream stream, DrawId draw) noexcept;
void fill_bytes(Stream stream, DrawId first_block,
                std::span<std::byte> out) noexcept;

} // namespace rund::host::random
