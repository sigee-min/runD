#include "src/host/net/test/ticket.hpp"
#include "local.hpp"
#include "src/host/net/payload/algorithm.hpp"
#include <rund/net/bytes.hpp>
#include <rund/net/ready/ticket.hpp>
#include <rund/net/vectored.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <limits>

#include <sys/types.h>

namespace {

struct IdentityProbe final {
  class Sequence final {
  public:
    explicit Sequence(IdentityProbe &owner) noexcept : owner_(&owner) {}

    void Append(const std::span<const std::byte> bytes) noexcept {
      owner_->hashed_bytes += bytes.size();
    }

    [[nodiscard]] rund::StableHash Finish() const noexcept {
      return {.value = 2u};
    }

  private:
    IdentityProbe *owner_;
  };

  template <typename Slice>
  void Visit(const Slice &, const std::size_t) noexcept {
    ++descriptor_visits;
  }

  [[nodiscard]] rund::StableHash Hash(const std::byte *,
                                      const std::size_t bytes) noexcept {
    ++identity_calls;
    hashed_bytes += bytes;
    return {.value = 1u};
  }

  [[nodiscard]] Sequence Begin() noexcept {
    ++stream_calls;
    return Sequence{*this};
  }

  std::size_t identity_calls = 0u;
  std::size_t stream_calls = 0u;
  std::size_t descriptor_visits = 0u;
  std::size_t hashed_bytes = 0u;
};

[[nodiscard]] bool PayloadTraversalUsesOneCanonicalPass() {
  const std::array<std::byte, 7u> bytes{
      std::byte{'a'}, std::byte{'b'}, std::byte{'c'}, std::byte{'d'},
      std::byte{'e'}, std::byte{'f'}, std::byte{'g'}};
  const rund::node::NativeCallResult native{
      .value = 4,
      .state = rund::node::NativeCallState::Complete,
  };
  IdentityProbe contiguous{};
  (void)rund::net::payload_detail::Native(native, bytes.data(), bytes.size(),
                                          contiguous);
  VECTORED_ASSERT(contiguous.identity_calls == 1u);
  VECTORED_ASSERT(contiguous.stream_calls == 0u);
  VECTORED_ASSERT(contiguous.descriptor_visits == 0u);
  VECTORED_ASSERT(contiguous.hashed_bytes == 4u);

  const std::array<rund::net::batch::Slice, 4u> slices{
      rund::net::batch::Slice{.data = bytes.data(), .size = 2u},
      rund::net::batch::Slice{.data = bytes.data() + 2u, .size = 0u},
      rund::net::batch::Slice{.data = bytes.data() + 2u, .size = 3u},
      rund::net::batch::Slice{.data = bytes.data() + 5u, .size = 2u},
  };
  IdentityProbe vectored{};
  (void)rund::net::payload_detail::Prefix(
      std::span<const rund::net::batch::Slice>{slices}, 4u, vectored);
  VECTORED_ASSERT(vectored.identity_calls == 0u);
  VECTORED_ASSERT(vectored.stream_calls == 1u);
  VECTORED_ASSERT(vectored.descriptor_visits == 3u);
  VECTORED_ASSERT(vectored.hashed_bytes == 4u);
  return true;
}

} // namespace

bool NetVectoredRejectsImpossibleSliceMetadata() {
  VECTORED_ASSERT(PayloadTraversalUsesOneCanonicalPass());
  SocketPair pair{};
  VECTORED_ASSERT(OpenSocketPair(pair));

  std::array<std::byte, 1u> one_byte{std::byte{'x'}};
  const std::size_t ssize_max =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::array<rund::net::batch::Slice, 1u> oversized_send{
      rund::net::batch::Slice{.data = one_byte.data(), .size = ssize_max + 1u},
  };
  const std::array<rund::net::batch::Buffer, 1u> oversized_recv{
      rund::net::batch::Buffer{.data = one_byte.data(), .size = ssize_max + 1u},
  };
  const std::array<rund::net::batch::Slice, 2u> overflowing_send{
      rund::net::batch::Slice{.data = one_byte.data(), .size = ssize_max},
      rund::net::batch::Slice{.data = one_byte.data(), .size = 1u},
  };
  const std::array<rund::net::batch::Buffer, 2u> overflowing_recv{
      rund::net::batch::Buffer{.data = one_byte.data(), .size = ssize_max},
      rund::net::batch::Buffer{.data = one_byte.data(), .size = 1u},
  };

  rund::net::SendResult rejected_oversized_send{};
  rund::net::ReceiveResult rejected_oversized_recv{};
  rund::net::SendResult rejected_overflowing_send{};
  rund::net::ReceiveResult rejected_overflowing_recv{};
  struct Case final {
    SocketPair *pair;
    const std::array<rund::net::batch::Slice, 1u> *oversized_send;
    const std::array<rund::net::batch::Buffer, 1u> *oversized_recv;
    const std::array<rund::net::batch::Slice, 2u> *overflowing_send;
    const std::array<rund::net::batch::Buffer, 2u> *overflowing_recv;
    rund::net::SendResult *rejected_oversized_send;
    rund::net::ReceiveResult *rejected_oversized_recv;
    rund::net::SendResult *rejected_overflowing_send;
    rund::net::ReceiveResult *rejected_overflowing_recv;
  } test{&pair,
         &oversized_send,
         &oversized_recv,
         &overflowing_send,
         &overflowing_recv,
         &rejected_oversized_send,
         &rejected_oversized_recv,
         &rejected_overflowing_send,
         &rejected_overflowing_recv};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 4u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("net-vectored-impossible-size", [state = &test] {
              *state->rejected_oversized_send = rund::net::batch::send(
                  rund::node::test::net::ticket(
                      state->pair->left.view(),
                      rund::net::ready::Interest::Writable),
                  *state->oversized_send);
              *state->rejected_oversized_recv = rund::net::batch::receive(
                  rund::node::test::net::ticket(
                      state->pair->right.view(),
                      rund::net::ready::Interest::Readable),
                  *state->oversized_recv);
              *state->rejected_overflowing_send = rund::net::batch::send(
                  rund::node::test::net::ticket(
                      state->pair->left.view(),
                      rund::net::ready::Interest::Writable),
                  *state->overflowing_send);
              *state->rejected_overflowing_recv = rund::net::batch::receive(
                  rund::node::test::net::ticket(
                      state->pair->right.view(),
                      rund::net::ready::Interest::Readable),
                  *state->overflowing_recv);
            });
        joined = rund::task::join(task);
      });

  VECTORED_ASSERT(report.ok());
  VECTORED_ASSERT(joined.ok());
  VECTORED_ASSERT(!rejected_oversized_send.ok());
  VECTORED_ASSERT(rejected_oversized_send.code() ==
                  rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!rejected_oversized_recv.ok());
  VECTORED_ASSERT(rejected_oversized_recv.code() ==
                  rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!rejected_overflowing_send.ok());
  VECTORED_ASSERT(rejected_overflowing_send.code() ==
                  rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(!rejected_overflowing_recv.ok());
  VECTORED_ASSERT(rejected_overflowing_recv.code() ==
                  rund::ReasonCode::TaskInvalid);
  VECTORED_ASSERT(report.events().empty());
  return true;
}
