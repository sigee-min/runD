#pragma once
#include <rund/task/channel.hpp>

inline rund::task::Task<void>
ChannelSendAwait(rund::task::channel<int>* const channel,
                 std::atomic<std::uint32_t>* const after_await) {
  const rund::task::Status sent = co_await channel->send(7);
  (void)sent;
  after_await->fetch_add(1u, std::memory_order_release);
}

inline rund::task::Task<void>
ChannelRecvAwait(rund::task::channel<int>* const channel,
                 std::atomic<std::uint32_t>* const after_await,
                 int* const value) {
  rund::task::ReceiveResult<int> received = co_await channel->recv();
  TEST_ASSERT(received);
  *value = *received;
  after_await->fetch_add(1u, std::memory_order_release);
}

inline rund::task::Task<void> ReceiveOne(rund::task::channel<int>* const channel) {
  (void)co_await channel->recv();
}

inline rund::task::Task<void> SendOne(rund::task::channel<int>* const channel, const int value) {
  (void)co_await channel->send(value);
}

inline rund::task::Task<void> DiscardChannelOps(rund::task::channel<int>* const channel,
                             std::atomic<std::uint32_t>* const completed) {
  (void)channel->send(19);
  (void)channel->recv();
  completed->fetch_add(1u, std::memory_order_release);
  co_return;
}
