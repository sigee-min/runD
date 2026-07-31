#include "src/runtime/task/scheduler/hash.hpp"
#include "test/assert.hpp"

#include <cstdint>

namespace {

consteval ::rund::detail::task::StatStorage SeededStats() {
  ::rund::detail::task::StatStorage stats{};
  ::rund::detail::task::Stat(stats, ::rund::detail::task::StatSlot::TraceHash) =
      ::rund::detail::task::kTraceHashSeed;
  return stats;
}

consteval std::uint64_t OperationHash() {
  auto stats = SeededStats();
  ::rund::node::HashOperation(
      stats, 1u, ::rund::detail::task::OperationKind::ChannelSend, 2u, 3u, 4u,
      5u, -6, static_cast<short>(-7), static_cast<short>(8), -9, 10u, 11u, 12u,
      13u, 14u, 15u, 16u, 17u, 18u, ::rund::ReasonCode::TaskFailed,
      ::rund::ReasonCode::Ok);
  return ::rund::detail::task::Stat(stats,
                                    ::rund::detail::task::StatSlot::TraceHash);
}

consteval std::uint64_t ObservationHash() {
  auto stats = SeededStats();
  ::rund::node::HashObservation(
      stats, ::rund::task::Observation{
                 .sequence = 21u,
                 .kind = ::rund::task::ObservationKind::IoInvalid,
                 .task_id = 22u,
                 .wait_id = 23u,
                 .fd = -24,
                 .interest = static_cast<short>(-25),
                 .revents = static_cast<short>(26),
                 .deadline_ns = -27,
                 .reason_code = ::rund::ReasonCode::TaskFailed,
             });
  return ::rund::detail::task::Stat(stats,
                                    ::rund::detail::task::StatSlot::TraceHash);
}

consteval std::uint64_t HostHash() {
  auto stats = SeededStats();
  ::rund::node::HashHost(stats, 0x0123456789abcdefull);
  return ::rund::detail::task::Stat(stats,
                                    ::rund::detail::task::StatSlot::TraceHash);
}

static_assert(OperationHash() == 0x1226e20f0f04887eull);
static_assert(ObservationHash() == 0xd76f34d7a54aef66ull);
static_assert(HostHash() == 0x932d7a3b77024088ull);
static_assert(OperationHash() != ObservationHash());
static_assert(OperationHash() != HostHash());
static_assert(ObservationHash() != HostHash());

} // namespace

int RunRuntimeTaskHashContract() {
  TEST_ASSERT(OperationHash() == 0x1226e20f0f04887eull);
  TEST_ASSERT(ObservationHash() == 0xd76f34d7a54aef66ull);
  TEST_ASSERT(HostHash() == 0x932d7a3b77024088ull);
  return 0;
}
