#pragma once

#include <rund/host/hash.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::host_detail {

inline constexpr std::uint64_t kStableByteHashSeed =
    0x72756e642e627974ull; // "rund.byt"
inline constexpr std::uint64_t kNullByteHashSeed =
    0x72756e642e6e756cull; // "rund.nul"

class StableByteHasher final {
public:
  StableByteHasher() noexcept;
  ~StableByteHasher() noexcept;

  StableByteHasher(const StableByteHasher &) = delete;
  StableByteHasher &operator=(const StableByteHasher &) = delete;
  StableByteHasher(StableByteHasher &&) = delete;
  StableByteHasher &operator=(StableByteHasher &&) = delete;

  void Append(std::span<const std::byte> bytes) noexcept;
  [[nodiscard]] ::rund::StableHash Finish() const noexcept;

private:
  alignas(64) std::array<std::byte, 576u> state_{};
};

[[nodiscard]] ::rund::StableHash StableByteHash(const std::byte *data,
                                              std::size_t size) noexcept;

} // namespace rund::node::host_detail
