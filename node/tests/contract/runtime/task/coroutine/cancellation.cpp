#include "local.hpp"
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/channel.hpp>

namespace rund::node::test_contract::coroutine {

rund::task::Task<void>
ObserveClosedRecv(rund::task::channel<int> *const channel,
                  std::atomic<std::uint32_t> *const after_await,
                  rund::ReasonCode *const code) {
  rund::task::ReceiveResult<int> received = co_await channel->recv();
  *code = received.code();
  after_await->fetch_add(1u, std::memory_order_release);
}

rund::task::Task<void>
ObserveChannelCapacity(rund::task::channel<int> *const channel,
                       std::atomic<std::uint32_t> *const after_await,
                       rund::ReasonCode *const code) {
  const rund::task::Status sent = co_await channel->send(3);
  *code = sent.code();
  after_await->fetch_add(1u, std::memory_order_release);
}

rund::task::Task<void>
ObserveJoin(const rund::task::Handle handle,
            std::atomic<std::uint32_t> *const after_await,
            rund::ReasonCode *const code) {
  const rund::task::Status joined = co_await handle;
  *code = joined.code();
  after_await->fetch_add(1u, std::memory_order_release);
}

int RunRuntimeTaskCoroutineCancellationContract() {
  std::atomic<std::uint32_t> after_closed_recv{0u};
  rund::ReasonCode closed_recv_code = rund::ReasonCode::Ok;
  rund::task::Status closed_recv_join{};
  bool closed_recv_handle_valid = false;
  const rund::Session::Result closed_recv_report = rund::run(
      rund::SessionConfig{
          .id = 796u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        auto channel = rund::task::channel<int>::make(0u);
        (void)channel.close();
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-closed-recv",
            ObserveClosedRecv(&channel, &after_closed_recv, &closed_recv_code));
        closed_recv_handle_valid = static_cast<bool>(task);
        closed_recv_join = rund::task::join(task);
      });
  TEST_ASSERT(closed_recv_report.ok());
  TEST_ASSERT(closed_recv_handle_valid);
  TEST_ASSERT(closed_recv_join.ok());
  TEST_ASSERT(after_closed_recv.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(closed_recv_code == rund::ReasonCode::ChannelClosed);

  std::atomic<std::uint32_t> after_channel_capacity{0u};
  rund::ReasonCode channel_capacity_code = rund::ReasonCode::Ok;
  rund::task::Status channel_capacity_join{};
  bool channel_capacity_handle_valid = false;
  const rund::Session::Result channel_capacity_report = rund::run(
      rund::SessionConfig{
          .id = 797u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .channel_capacity = 1u,
                  .channel_buffer_capacity = 0u,
                  .channel_wait_capacity = 0u,
              },
      },
      [&] {
        auto channel = rund::task::channel<int>::make(0u);
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-channel-capacity",
            ObserveChannelCapacity(&channel, &after_channel_capacity,
                                   &channel_capacity_code));
        channel_capacity_handle_valid = static_cast<bool>(task);
        channel_capacity_join = rund::task::join(task);
      });
  TEST_ASSERT(channel_capacity_report.ok());
  TEST_ASSERT(channel_capacity_handle_valid);
  TEST_ASSERT(channel_capacity_join.ok());
  TEST_ASSERT(after_channel_capacity.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(channel_capacity_code ==
              rund::ReasonCode::ChannelWaitCapacityExceeded);

  std::atomic<std::uint32_t> after_stale_join{0u};
  rund::ReasonCode stale_join_code = rund::ReasonCode::Ok;
  rund::task::Status stale_join{};
  bool stale_join_handle_valid = false;
  const rund::Session::Result stale_join_report = rund::run(
      rund::SessionConfig{
          .id = 798u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle child =
            rund::task::spawn("coroutine-stale-child", [] {});
        (void)rund::task::join(child);
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-stale-join",
            ObserveJoin(child, &after_stale_join, &stale_join_code));
        stale_join_handle_valid = static_cast<bool>(task);
        stale_join = rund::task::join(task);
      });
  TEST_ASSERT(stale_join_report.ok());
  TEST_ASSERT(stale_join_handle_valid);
  TEST_ASSERT(stale_join.ok());
  TEST_ASSERT(after_stale_join.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(stale_join_code == rund::ReasonCode::TaskHandleUnknown);
  return 0;
}

} // namespace rund::node::test_contract::coroutine
