#include "src/runtime/task/scheduler/task/frame.hpp"
#include "allocation.hpp"
#include "local.hpp"
#include "src/runtime/task/scheduler/task/frame/bind.hpp"
#include <rund/task/await.hpp>
#include <rund/task/channel.hpp>

#include <array>

namespace rund::node::test_contract::coroutine {
namespace {

// This fixture exercises arena admission itself. Without an out-of-line
// boundary an optimizing compiler may legally embed the coroutine frame in
// the caller and elide promise allocation, which would test allocation
// elision rather than the runtime frame authority.
[[gnu::noinline]] rund::task::Task<void> FrameComplete() { co_return; }

[[gnu::noinline]] rund::task::Task<void>
CompactChannelWait(rund::task::channel<std::uint32_t> *const gate) {
  (void)co_await gate->recv();
}

[[gnu::noinline]] rund::task::Task<void> OversizedFrame() {
  volatile std::byte state[1024u]{};
  const rund::task::Status yielded = co_await rund::task::yield();
  (void)yielded;
  for (std::size_t index = 0u; index < 1024u; ++index) {
    state[index] = std::byte{1u};
  }
  const std::byte observed = state[1023u];
  (void)observed;
}

[[nodiscard]] bool CompleteAndDestroy(rund::task::Task<void> task) noexcept {
  ::rund::detail::task::CoroutineStart frame =
      ::rund::detail::task::TakeCoroutine(std::move(task));
  if (frame.frame == nullptr || frame.ops == nullptr) {
    return false;
  }
  frame.frame.resume();
  const bool completed = frame.frame.done();
  ::rund::detail::task::DestroyCoroutine(frame);
  const bool retired = frame.frame == nullptr && frame.ops == nullptr &&
                       frame.code == ReasonCode::TaskInvalid;
  ::rund::detail::task::DestroyCoroutine(frame);
  return completed && retired && frame.frame == nullptr && frame.ops == nullptr;
}

int CheckArenaLeaseContract() {
  FrameArena arena{};
  TEST_ASSERT(
      arena
          .configure(FrameLimits{.capacity = 1u, .bytes = 64u, .alignment = 8u})
          .code() == ReasonCode::TaskFrameLimitsInvalid);
  TEST_ASSERT(arena.configure(
      FrameLimits{.capacity = 1u, .bytes = 64u, .alignment = 16u}));
  TEST_ASSERT(arena.stats().resident_slots == 0u);
  TEST_ASSERT(arena.stats().resident_bytes == 0u);
  TEST_ASSERT(!arena.acquire(8u, 3u));
  TEST_ASSERT(arena.code() == ReasonCode::TaskFrameAlignment);
  TEST_ASSERT(!arena.acquire(65u, 8u));
  TEST_ASSERT(arena.code() == ReasonCode::TaskFrameTooLarge);

  const FrameLease first = arena.acquire(32u, 8u);
  TEST_ASSERT(first);
  TEST_ASSERT(arena.stats().resident_slots == 1u);
  TEST_ASSERT(arena.stats().resident_bytes == 80u);
  TEST_ASSERT(!arena.acquire(8u, 8u));
  TEST_ASSERT(arena.code() == ReasonCode::TaskFrameCapacity);
  arena.release(first);

  const FrameLease reused = arena.acquire(32u, 8u);
  TEST_ASSERT(reused);
  TEST_ASSERT(reused.slot == first.slot);
  TEST_ASSERT(reused.generation != first.generation);
  arena.release(first);
  TEST_ASSERT(arena.stats().live == 1u);
  arena.release(reused);
  arena.release(reused);

  const FrameStats stats = arena.stats();
  TEST_ASSERT(stats.live == 0u);
  TEST_ASSERT(stats.high_water == 1u);
  TEST_ASSERT(stats.allocations == 2u);
  TEST_ASSERT(stats.reuses == 1u);
  TEST_ASSERT(stats.failures == 5u);
  return 0;
}

int CheckArenaCapacityContract() {
  FrameArena arena{};
  TEST_ASSERT(arena.configure(
      FrameLimits{.capacity = 4u, .bytes = 1024u, .alignment = 64u}));
  std::array<FrameLease, 4u> leases{};
  for (FrameLease &lease : leases) {
    lease = arena.acquire(1024u, 64u);
    TEST_ASSERT(lease);
  }
  TEST_ASSERT(!arena.acquire(1u, 64u));
  TEST_ASSERT(arena.code() == ReasonCode::TaskFrameCapacity);
  for (const FrameLease lease : leases) {
    arena.release(lease);
  }
  const FrameStats stats = arena.stats();
  TEST_ASSERT(stats.live == 0u);
  TEST_ASSERT(stats.high_water == 4u);
  TEST_ASSERT(stats.allocations == 4u);
  TEST_ASSERT(stats.failures == 1u);
  return 0;
}

int CheckArenaTierContract() {
  FrameArena arena{};
  TEST_ASSERT(arena.configure(
      FrameLimits{.capacity = 2u, .bytes = 2048u, .alignment = 64u}));
  const FrameLease compact = arena.acquire(160u, 64u);
  const FrameLease wide = arena.acquire(1024u, 64u);
  TEST_ASSERT(compact);
  TEST_ASSERT(wide);
  TEST_ASSERT(compact.tier == 0u);
  TEST_ASSERT(wide.tier == 1u);
  TEST_ASSERT(!arena.acquire(1u, 64u));
  TEST_ASSERT(arena.code() == ReasonCode::TaskFrameCapacity);
  arena.release(compact);
  arena.release(wide);
  runtime_task_allocation::Start();
  const FrameLease warm_wide = arena.acquire(1024u, 64u);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm_wide);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  arena.release(warm_wide);
  const FrameStats stats = arena.stats();
  TEST_ASSERT(stats.live == 0u);
  TEST_ASSERT(stats.high_water == 2u);
  TEST_ASSERT(stats.compact_allocations == 1u);
  TEST_ASSERT(stats.wide_allocations == 2u);
  TEST_ASSERT(stats.reuses == 1u);
  TEST_ASSERT(stats.failures == 1u);
  return 0;
}

int CheckPromiseAdmissionContract() {
  FrameArena arena{};
  TEST_ASSERT(arena.configure(
      FrameLimits{.capacity = 1u, .bytes = 64u, .alignment = 64u}));
  bool rejected = false;
  {
    BindFrameArena bind{arena};
    rund::task::Task<void> task = OversizedFrame();
    rejected = !task && task.code() == ReasonCode::TaskFrameTooLarge;
  }
  TEST_ASSERT(rejected);
  TEST_ASSERT(arena.stats().failures == 1u);

  FrameArena outer_arena{};
  TEST_ASSERT(outer_arena.configure(
      FrameLimits{.capacity = 1u, .bytes = 4096u, .alignment = 64u}));
  FrameArena other{};
  TEST_ASSERT(other.configure(
      FrameLimits{.capacity = 1u, .bytes = 4096u, .alignment = 64u}));
  bool mismatch_rejected = false;
  bool outer_restored = false;
  {
    BindFrameArena outer{outer_arena};
    {
      BindFrameArena incompatible{other};
      rund::task::Task<void> task = FrameComplete();
      mismatch_rejected =
          !task && task.code() == ReasonCode::TaskFrameRuntimeMismatch;
    }
    rund::task::Task<void> task = FrameComplete();
    outer_restored = static_cast<bool>(task);
  }
  TEST_ASSERT(mismatch_rejected);
  TEST_ASSERT(outer_restored);

  FrameArena compact{};
  TEST_ASSERT(compact.configure(
      FrameLimits{.capacity = 1u, .bytes = 224u, .alignment = 16u}));
  {
    BindFrameArena bind{compact};
    rund::task::Task<void> task = CompactChannelWait(nullptr);
    TEST_ASSERT(task);
    TEST_ASSERT(::rund::detail::task::frame::Bytes(
                    ::rund::detail::task::CoroutineAddress(task)) <= 224u);
  }
  return 0;
}

int CheckEscapedFrameLifetime() {
  rund::task::Task<void> escaped{};
  {
    FrameArena arena{};
    TEST_ASSERT(arena.configure(
        FrameLimits{.capacity = 1u, .bytes = 4096u, .alignment = 64u}));
    BindFrameArena bind{arena};
    escaped = FrameComplete();
    TEST_ASSERT(escaped);
  }
  TEST_ASSERT(CompleteAndDestroy(std::move(escaped)));
  return 0;
}

} // namespace

int RunRuntimeTaskCoroutineFrameContract() {
  TEST_ASSERT(CheckArenaLeaseContract() == 0);
  TEST_ASSERT(CheckArenaCapacityContract() == 0);
  TEST_ASSERT(CheckArenaTierContract() == 0);
  TEST_ASSERT(CheckPromiseAdmissionContract() == 0);
  TEST_ASSERT(CheckEscapedFrameLifetime() == 0);
  FrameArena arena{};
  TEST_ASSERT(arena.configure(
      FrameLimits{.capacity = 4u, .bytes = 16u * 1024u, .alignment = 64u}));
  std::uint64_t warm_allocations = 0u;
  {
    BindFrameArena bind{arena};
    rund::task::Task<void> first = FrameComplete();
    TEST_ASSERT(first);
    TEST_ASSERT(CompleteAndDestroy(std::move(first)));

    runtime_task_allocation::Start();
    rund::task::Task<void> warm = FrameComplete();
    runtime_task_allocation::Stop();
    warm_allocations = runtime_task_allocation::Count();
    TEST_ASSERT(warm);
    TEST_ASSERT(CompleteAndDestroy(std::move(warm)));
  }
  TEST_ASSERT(warm_allocations == 0u);
  const FrameStats stats = arena.stats();
  TEST_ASSERT(stats.resident_slots == 4u);
  TEST_ASSERT(stats.resident_bytes == 1024u);
  TEST_ASSERT(stats.live == 0u);
  TEST_ASSERT(stats.high_water == 1u);
  TEST_ASSERT(stats.allocations == 2u);
  TEST_ASSERT(stats.reuses == 1u);
  TEST_ASSERT(stats.failures == 0u);
  return 0;
}

} // namespace rund::node::test_contract::coroutine

int RunRuntimeTaskCoroutineFrameContract() {
  return rund::node::test_contract::coroutine::
      RunRuntimeTaskCoroutineFrameContract();
}
