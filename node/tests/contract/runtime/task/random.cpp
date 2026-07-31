#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/host.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace {

constexpr std::uint64_t kDefaultRunSeed = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint64_t kEntitySeed = 0xB9462B77FDCC3189ull;
constexpr std::uint64_t kDrawOneU64 = 0xBA0DF72189142F3Eull;
constexpr std::uint32_t kDrawOneU32 = 0xBA0DF721u;
constexpr std::uint32_t kDrawFiveUnitF32Bits = 0x3F479406u;
constexpr std::uint64_t kUint64Max =
    std::numeric_limits<std::uint64_t>::max();

constexpr std::array<std::byte, 19u> kExpectedBytes{
    std::byte{0xe2u}, std::byte{0x97u}, std::byte{0x4bu},
    std::byte{0x00u}, std::byte{0xf7u}, std::byte{0xc2u},
    std::byte{0x10u}, std::byte{0x19u}, std::byte{0xefu},
    std::byte{0x00u}, std::byte{0xbcu}, std::byte{0xc2u},
    std::byte{0x53u}, std::byte{0x45u}, std::byte{0x7du},
    std::byte{0x80u}, std::byte{0xc6u}, std::byte{0x33u},
    std::byte{0xb5u}};

constexpr std::array<std::byte, 19u> kExpectedWrappedBytes{
    std::byte{0x96u}, std::byte{0x78u}, std::byte{0x82u},
    std::byte{0xbbu}, std::byte{0xf1u}, std::byte{0xa1u},
    std::byte{0x55u}, std::byte{0xe6u}, std::byte{0x42u},
    std::byte{0x80u}, std::byte{0x56u}, std::byte{0x38u},
    std::byte{0x0fu}, std::byte{0x9du}, std::byte{0x5cu},
    std::byte{0xd9u}, std::byte{0x53u}, std::byte{0xb9u},
    std::byte{0xd6u}};

}  // namespace

int RunRuntimeTaskRandomContract() {
  const auto world = rund::host::random::stream(
      rund::host::random::StreamId{.value = 17u});
  const auto entity = rund::host::random::split(
      world, rund::host::random::StreamId{.value = 99u});
  TEST_ASSERT(world.seed == kDefaultRunSeed);
  TEST_ASSERT(world.id == 17u);
  TEST_ASSERT(entity.seed == kEntitySeed);
  TEST_ASSERT(entity.id == 99u);

  const std::uint64_t first =
      rund::host::random::u64(entity, rund::host::random::DrawId{.value = 1u});
  const std::uint64_t second =
      rund::host::random::u64(entity, rund::host::random::DrawId{.value = 1u});
  TEST_ASSERT(first == second);
  TEST_ASSERT(first == kDrawOneU64);
  TEST_ASSERT(rund::host::random::u32(
                  entity, rund::host::random::DrawId{.value = 1u}) ==
              kDrawOneU32);

  const std::uint64_t other =
      rund::host::random::u64(entity, rund::host::random::DrawId{.value = 2u});
  TEST_ASSERT(first != other);
  TEST_ASSERT(rund::host::random::below(
                  entity, rund::host::random::DrawId{.value = 5u}, 7u) == 2u);
  TEST_ASSERT(rund::host::random::range(
                  entity, rund::host::random::DrawId{.value = 5u}, 10u,
                  20u) == 16u);
  TEST_ASSERT(std::bit_cast<std::uint32_t>(rund::host::random::unit_f32(
                  entity, rund::host::random::DrawId{.value = 5u})) ==
              kDrawFiveUnitF32Bits);

  for (std::uint64_t index = 0u; index < 1024u; ++index) {
    TEST_ASSERT(rund::host::random::below(
                    entity,
                    rund::host::random::DrawId{.value = index},
                    7u) < 7u);
    const std::uint64_t ranged = rund::host::random::range(
        entity, rund::host::random::DrawId{.value = index}, 10u, 20u);
    TEST_ASSERT(ranged >= 10u);
    TEST_ASSERT(ranged < 20u);
    const float unit = rund::host::random::unit_f32(
        entity, rund::host::random::DrawId{.value = index});
    TEST_ASSERT(unit >= 0.0f);
    TEST_ASSERT(unit < 1.0f);
  }

  std::array<std::byte, 19u> bytes_a{};
  std::array<std::byte, 19u> bytes_b{};
  rund::host::random::fill_bytes(
      entity, rund::host::random::DrawId{.value = 3u}, bytes_a);
  rund::host::random::fill_bytes(
      entity, rund::host::random::DrawId{.value = 3u}, bytes_b);
  TEST_ASSERT(bytes_a == bytes_b);
  TEST_ASSERT(bytes_a == kExpectedBytes);

  TEST_ASSERT(rund::host::random::below(
                  entity, rund::host::random::DrawId{.value = 5u}, 0u) == 0u);
  TEST_ASSERT(rund::host::random::below(
                  entity, rund::host::random::DrawId{.value = 5u}, 1u) == 0u);
  TEST_ASSERT(rund::host::random::below(
                  entity, rund::host::random::DrawId{.value = 5u},
                  kUint64Max) < kUint64Max);
  TEST_ASSERT(rund::host::random::range(
                  entity, rund::host::random::DrawId{.value = 5u}, 42u,
                  42u) == 42u);
  TEST_ASSERT(rund::host::random::range(
                  entity, rund::host::random::DrawId{.value = 5u}, 100u,
                  20u) == 100u);

  std::array<std::byte, 1u> zero_length{std::byte{0xabu}};
  rund::host::random::fill_bytes(
      entity, rund::host::random::DrawId{.value = 3u},
      std::span<std::byte>{zero_length.data(), 0u});
  TEST_ASSERT(zero_length[0u] == std::byte{0xabu});

  std::array<std::byte, 19u> wrapped_bytes{};
  rund::host::random::fill_bytes(
      entity, rund::host::random::DrawId{.value = kUint64Max - 1u},
      wrapped_bytes);
  TEST_ASSERT(wrapped_bytes == kExpectedWrappedBytes);

  const auto run_seeded_draw = [](const std::uint64_t seed,
                                  const std::uint32_t workers,
                                  const std::uint32_t task_workers)
      -> std::uint64_t {
    std::uint64_t value = 0u;
    const rund::Session::Result result = rund::run(
        rund::SessionConfig{
          .id = 33u,
          .workers = workers,
          .scheduler = {
            .task_workers = task_workers,
          },
          .random_seed = seed,
        },
        [&]() {
          const auto stream = rund::host::random::stream(
              rund::host::random::StreamId{.value = 77u});
          value = rund::host::random::u64(
              stream, rund::host::random::DrawId{.value = 9u});
        });
    TEST_ASSERT(result.ok());
    return value;
  };

  const std::uint64_t seeded_single_worker =
      run_seeded_draw(0x1234u, 1u, 1u);
  const std::uint64_t seeded_multi_worker =
      run_seeded_draw(0x1234u, 2u, 2u);
  const std::uint64_t differently_seeded =
      run_seeded_draw(0x5678u, 1u, 1u);
  TEST_ASSERT(seeded_single_worker == seeded_multi_worker);
  TEST_ASSERT(seeded_single_worker != differently_seeded);

  return 0;
}
