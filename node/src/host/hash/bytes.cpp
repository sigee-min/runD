#include "bytes.hpp"

#define XXH_PRIVATE_API
#include "../vendor/xxhash/xxhash.h"

#include <array>
#include <new>
#include <type_traits>

namespace rund::node::host_detail {
namespace {

static_assert(sizeof(XXH3_state_t) == 576u);
static_assert(alignof(XXH3_state_t) == 64u);
static_assert(std::is_trivially_destructible_v<XXH3_state_t>);

[[nodiscard]] XXH3_state_t *State(
    std::array<std::byte, 576u> &storage) noexcept {
  return reinterpret_cast<XXH3_state_t *>(storage.data());
}

[[nodiscard]] const XXH3_state_t *State(
    const std::array<std::byte, 576u> &storage) noexcept {
  return reinterpret_cast<const XXH3_state_t *>(storage.data());
}

[[nodiscard]] ::rund::StableHash NullByteHash(const std::size_t size) noexcept {
  std::array<std::byte, sizeof(std::uint64_t)> framing{};
  const std::uint64_t value = static_cast<std::uint64_t>(size);
  for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
    framing[shift / 8u] =
        static_cast<std::byte>((value >> shift) & 0xffu);
  }
  return ::rund::StableHash{
      .value = XXH3_64bits_withSeed(framing.data(), framing.size(),
                                    kNullByteHashSeed)};
}

} // namespace

StableByteHasher::StableByteHasher() noexcept {
  ::new (static_cast<void *>(state_.data())) XXH3_state_t{};
  (void)XXH3_64bits_reset_withSeed(State(state_), kStableByteHashSeed);
}

StableByteHasher::~StableByteHasher() noexcept = default;

void StableByteHasher::Append(
    const std::span<const std::byte> bytes) noexcept {
  if (!bytes.empty()) {
    (void)XXH3_64bits_update(State(state_), bytes.data(), bytes.size());
  }
}

::rund::StableHash StableByteHasher::Finish() const noexcept {
  return ::rund::StableHash{.value = XXH3_64bits_digest(State(state_))};
}

::rund::StableHash StableByteHash(const std::byte *const data,
                                const std::size_t size) noexcept {
  if (data == nullptr && size != 0u) {
    return NullByteHash(size);
  }
  static constexpr std::byte kEmpty{};
  const std::byte *const bytes = data == nullptr ? &kEmpty : data;
  return ::rund::StableHash{
      .value = XXH3_64bits_withSeed(bytes, size, kStableByteHashSeed)};
}

} // namespace rund::node::host_detail
