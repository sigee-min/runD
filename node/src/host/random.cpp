#include <rund/host/random.hpp>

#include "../runtime/task/scheduler/host.hpp"

namespace rund::host::random {
namespace {

constexpr std::uint64_t kStreamSalt = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kDrawSalt = 0xD1B54A32D192ED03ull;

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

} // namespace

RunSeed active_run_seed() noexcept {
  return ::rund::node::scheduler_host::RandomSeed();
}

Stream stream(const RunSeed seed, const StreamId id) noexcept {
  return Stream{.seed = seed.value, .id = id.value};
}

Stream stream(const StreamId id) noexcept {
  return stream(active_run_seed(), id);
}

Stream split(const Stream parent, const StreamId child) noexcept {
  return Stream{.seed = splitmix64(parent.seed ^ (parent.id * kStreamSalt)),
                .id = child.value};
}

std::uint64_t u64(const Stream stream, const DrawId draw) noexcept {
  return splitmix64(stream.seed ^ (stream.id * kStreamSalt) ^
                    (draw.value * kDrawSalt));
}

std::uint32_t u32(const Stream stream, const DrawId draw) noexcept {
  return static_cast<std::uint32_t>(u64(stream, draw) >> 32u);
}

std::uint64_t below(const Stream stream, const DrawId draw,
                    const std::uint64_t upper_exclusive) noexcept {
  if (upper_exclusive == 0u) {
    return 0u;
  }
  const std::uint64_t threshold = (0u - upper_exclusive) % upper_exclusive;
  std::uint64_t index = draw.value;
  for (;;) {
    const std::uint64_t value = u64(stream, DrawId{.value = index});
    if (value >= threshold) {
      return value % upper_exclusive;
    }
    ++index;
  }
}

std::uint64_t range(const Stream stream, const DrawId draw,
                    const std::uint64_t lower_inclusive,
                    const std::uint64_t upper_exclusive) noexcept {
  if (upper_exclusive <= lower_inclusive) {
    return lower_inclusive;
  }
  return lower_inclusive +
         below(stream, draw, upper_exclusive - lower_inclusive);
}

float unit_f32(const Stream stream, const DrawId draw) noexcept {
  const std::uint32_t bits = u32(stream, draw) >> 8u;
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

void fill_bytes(const Stream stream, const DrawId first_block,
                const std::span<std::byte> out) noexcept {
  std::size_t written = 0u;
  std::uint64_t block = 0u;
  while (written < out.size()) {
    std::uint64_t value =
        u64(stream, DrawId{.value = first_block.value + block});
    for (std::uint32_t byte = 0u; byte < 8u && written < out.size(); ++byte) {
      out[written++] = static_cast<std::byte>(value & 0xffu);
      value >>= 8u;
    }
    ++block;
  }
}

} // namespace rund::host::random
