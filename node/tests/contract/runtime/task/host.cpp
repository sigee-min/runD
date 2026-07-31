#include "test/assert.hpp"
#include "../../../../src/host/hash/bytes.hpp"

#include <rund/session.hpp>
#include <rund/host.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace host = rund::host;

constexpr std::uint64_t kHashAbc = 0x8fc30b5a6e55f267ull;
constexpr std::uint64_t kHashAbd = 0x81d07aa04248e69eull;
constexpr std::uint64_t kHashEmptyBytes = 0x3ddc06053ea59af8ull;
constexpr std::uint64_t kHashNullNonEmpty = 0x3e347c33b6e6942full;
constexpr std::uint64_t kHashRepresentativeEvent = 0x3f073af1655a1928ull;
constexpr std::uint64_t kHashEmptyEvents = 0x44bd2bd473ccf799ull;
constexpr std::array<std::size_t, 16u> kPatternSizes{
    1u, 3u, 4u, 7u, 8u, 9u, 15u, 16u,
    17u, 64u, 128u, 129u, 240u, 241u, 1500u, 65536u};
constexpr std::array<std::uint64_t, 16u> kPatternHashes{
    0x1753236354e7a5daull, 0xce76b59882d49bc3ull,
    0xd89b58273ade01b6ull, 0x48f999f97e519919ull,
    0x0d47364024cd7805ull, 0xc8426582f74ccaecull,
    0x59c18a3c94c2a102ull, 0x102a5c611b6180f1ull,
    0x69cb55de528635bcull, 0xc5808addc8d148cfull,
    0x6db28902fc8ec39eull, 0xa7f2513e5a86a884ull,
    0x8f8ac49de1880af4ull, 0x35ba56acd352fb30ull,
    0xb553d949bed8b707ull, 0x06e1add7351ff5adull};
constexpr std::array<std::size_t, 15u> kStreamParts{
    1u, 63u, 64u, 127u, 128u, 239u, 240u, 241u,
    255u, 256u, 257u, 1024u, 4096u, 17u, 3u};

static_assert(noexcept(::rund::host::hash_bytes(nullptr, 0u)));
static_assert(noexcept(::rund::host::hash_string(nullptr, 0u)));
static_assert(noexcept(::rund::host::hash_event(::rund::host::Event{})));
static_assert(noexcept(::rund::host::hash_events(std::span<const ::rund::host::Event>{})));

[[nodiscard]] bool EventMutationChangesHash(const ::rund::host::Event &baseline,
                                            const ::rund::host::Event &mutated) {
  return ::rund::host::hash_event(baseline).value != ::rund::host::hash_event(mutated).value;
}

} // namespace

int RunRuntimeTaskHostContract() {
  const char abc[] = "abc";
  const char abd[] = "abd";
  const ::rund::StableHash abc_hash = ::rund::host::hash_string(abc, 3u);
  const ::rund::StableHash abc_hash_again = ::rund::host::hash_string(abc, 3u);
  const ::rund::StableHash abd_hash = ::rund::host::hash_string(abd, 3u);
  TEST_ASSERT(abc_hash.value == kHashAbc);
  TEST_ASSERT(abc_hash_again.value == kHashAbc);
  TEST_ASSERT(abd_hash.value == kHashAbd);
  TEST_ASSERT(abc_hash.value != abd_hash.value);

  const std::array<std::byte, 0u> empty_bytes{};
  TEST_ASSERT(::rund::host::hash_bytes(empty_bytes.data(), empty_bytes.size()).value ==
              kHashEmptyBytes);
  TEST_ASSERT(::rund::host::hash_bytes(nullptr, 0u).value == kHashEmptyBytes);
  TEST_ASSERT(::rund::host::hash_string(nullptr, 0u).value == kHashEmptyBytes);
  TEST_ASSERT(::rund::host::hash_bytes(nullptr, 3u).value == kHashNullNonEmpty);
  TEST_ASSERT(::rund::host::hash_string(nullptr, 3u).value == kHashNullNonEmpty);

  std::vector<std::byte> pattern(65536u);
  for (std::size_t index = 0u; index < pattern.size(); ++index) {
    pattern[index] =
        static_cast<std::byte>((index * 131u + 17u) & 0xffu);
  }
  for (std::size_t row = 0u; row < kPatternSizes.size(); ++row) {
    const std::span<const std::byte> bytes{pattern.data(), kPatternSizes[row]};
    TEST_ASSERT(::rund::host::hash_bytes(bytes.data(), bytes.size()).value ==
                kPatternHashes[row]);
    rund::node::host_detail::StableByteHasher stream{};
    std::size_t offset = 0u;
    std::size_t part = 0u;
    while (offset < bytes.size()) {
      const std::size_t size =
          std::min(kStreamParts[part % kStreamParts.size()],
                   bytes.size() - offset);
      stream.Append(bytes.subspan(offset, size));
      stream.Append({});
      offset += size;
      ++part;
    }
    TEST_ASSERT(stream.Finish().value == kPatternHashes[row]);
  }

  const ::rund::host::Event event{
      .sequence = 11u,
      .kind = ::rund::host::EventKind::IoClose,
      .status = ::rund::host::Status::SyscallFailed,
      .task_id = 22u,
      .logical_time_ns = 33u,
      .stream_id = 44u,
      .draw_id = 55u,
      .host_handle_id = 66u,
      .offset = 77u,
      .requested_bytes = 88u,
      .completed_bytes = 99u,
      .native_errno = -5,
      .name_hash = ::rund::StableHash{.value = 0x1111222233334444ull},
      .path_hash = ::rund::StableHash{.value = 0x5555666677778888ull},
      .payload_hash = abc_hash};
  TEST_ASSERT(::rund::host::hash_event(event).value == kHashRepresentativeEvent);

  ::rund::host::Event mutated = event;
  mutated.sequence = 12u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.kind = ::rund::host::EventKind::IoWrite;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.status = ::rund::host::Status::ReplayMismatch;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.task_id = 23u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.logical_time_ns = 34u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.stream_id = 45u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.draw_id = 56u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.host_handle_id = 67u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.offset = 78u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.requested_bytes = 89u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.completed_bytes = 100u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.native_errno = -6;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.name_hash.value ^= 1u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.path_hash.value ^= 1u;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));
  mutated = event;
  mutated.payload_hash = abd_hash;
  TEST_ASSERT(EventMutationChangesHash(event, mutated));

  TEST_ASSERT(::rund::host::hash_events(std::span<const ::rund::host::Event>{}).value ==
              kHashEmptyEvents);
  ::rund::host::Event second_event = event;
  second_event.sequence = 12u;
  second_event.kind = ::rund::host::EventKind::IoWrite;
  second_event.completed_bytes = 101u;
  const std::array<::rund::host::Event, 1u> one_event{event};
  const std::array<::rund::host::Event, 2u> ordered_events{event, second_event};
  const std::array<::rund::host::Event, 2u> reversed_events{second_event, event};
  TEST_ASSERT(::rund::host::hash_events(ordered_events).value !=
              ::rund::host::hash_events(reversed_events).value);
  TEST_ASSERT(::rund::host::hash_events(ordered_events).value !=
              ::rund::host::hash_events(one_event).value);

  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(::rund::host::EventKind::None)} ==
              "none");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::RandomDraw)} == "random_draw");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::LogicalClockRead)} == "logical_clock_read");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::TimerSleep)} == "timer_sleep");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::IoReady)} == "io_ready");
  TEST_ASSERT(std::string_view{
                  ::rund::host::event_kind_name(::rund::host::EventKind::IoRead)} == "io_read");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::IoWrite)} == "io_write");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::IoClose)} == "io_close");
  TEST_ASSERT(std::string_view{
                  ::rund::host::event_kind_name(::rund::host::EventKind::EnvGet)} == "env_get");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::NetSocket)} == "net_socket");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::NetBind)} == "net_bind");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::NetListen)} == "net_listen");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::NetLocalAddress)} == "net_local_address");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  ::rund::host::EventKind::NetShutdown)} == "net_shutdown");
  TEST_ASSERT(std::string_view{::rund::host::event_kind_name(
                  static_cast<::rund::host::EventKind>(999u))} == "unknown");
  return 0;
}
