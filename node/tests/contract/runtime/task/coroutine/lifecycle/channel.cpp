#include <rund/task/api.hpp>
#include <rund/task/channel.hpp>
#include "../local.hpp"
#include "../allocation.hpp"

#include <algorithm>

namespace rund::node::test_contract::coroutine {

#include "channel/ops.hpp"

int CheckCoroutineChannelSend() {
  std::atomic<std::uint32_t> after{0u};
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 790u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
                [&] {
                  auto channel = rund::task::channel<int>::make(0u);
                  const rund::task::Handle sender = rund::task::spawn(
                      "coroutine-channel-send",
                      ChannelSendAwait(&channel, &after));
                  const rund::task::Handle receiver = rund::task::spawn(
                      "coroutine-channel-send-receiver", ReceiveOne(&channel));
                  handle_valid =
                      static_cast<bool>(sender) && static_cast<bool>(receiver);
                  joined = rund::task::join(sender, receiver);
                });
  return AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                     2u);
}

int CheckCoroutineChannelRecv() {
  std::atomic<std::uint32_t> after{0u};
  int value = 0;
  rund::task::Status joined{};
  bool handle_valid = false;
  const rund::Session::Result report =
      rund::run(rund::SessionConfig{
        .id = 791u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
                [&] {
                  auto channel = rund::task::channel<int>::make(0u);
                  const rund::task::Handle receiver = rund::task::spawn(
                      "coroutine-channel-recv",
                      ChannelRecvAwait(&channel, &after, &value));
                  const rund::task::Handle sender = rund::task::spawn(
                      "coroutine-channel-recv-sender", SendOne(&channel, 11));
                  handle_valid =
                      static_cast<bool>(receiver) && static_cast<bool>(sender);
                  joined = rund::task::join(receiver, sender);
                });
  if (AssertCoroutineAwaitSuccess(report, joined, handle_valid, after, 2u,
                                  2u) != 0) {
    return 1;
  }
  TEST_ASSERT(value == 11);
  return 0;
}

int CheckDiscardedChannelOps() {
  std::atomic<std::uint32_t> completed{0u};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .id = 795u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 2u,
          .ready_queue_capacity = 2u,
        },
      },
      [&] {
        auto channel = rund::task::channel<int>::make(0u);
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-channel-discard", DiscardChannelOps(&channel, &completed));
        joined = rund::task::join(task);
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(completed.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().channel_sends() == 0u);
  TEST_ASSERT(report.tasks().channel_recvs() == 0u);
  TEST_ASSERT(report.tasks().parked() == 0u);
  return 0;
}

int CheckCoroutineChannelRaii() {
  constexpr std::size_t rounds = 16u;
  bool made = true;
  std::uint64_t make_allocations = 0u;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .id = 799u,
        .workers = 2u,
        .scheduler = {
          .task_capacity = 2u,
          .ready_queue_capacity = 2u,
          .channel_capacity = 1u,
          .channel_buffer_capacity = 1u,
        },
      },
      [&] {
        for (std::size_t round = 0u; round < rounds; ++round) {
          runtime_task_allocation::Start();
          auto channel = rund::task::channel<int>::make(0u);
          runtime_task_allocation::Stop();
          make_allocations =
              std::max(make_allocations, runtime_task_allocation::Count());
          made = made && static_cast<bool>(channel);
        }
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(made);
  TEST_ASSERT(make_allocations == 1u);
  TEST_ASSERT(report.tasks().channel_closes() == rounds);
  return 0;
}

}  // namespace rund::node::test_contract::coroutine
