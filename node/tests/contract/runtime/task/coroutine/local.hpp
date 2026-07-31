#pragma once

#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/net/socket.hpp>
#include <rund/session.hpp>
#include <rund/task/channel.hpp>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unistd.h>

namespace rund::node::test_contract::coroutine {

rund::task::Task<void> CompleteOnWorker(std::atomic<std::uint32_t> *body_runs,
                                        std::thread::id *body_thread);
rund::task::Task<void> YieldOnce(std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> SleepOnce(std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> ReadyIoAwait(rund::host::io::FdView read_fd,
                                    std::atomic<std::uint32_t> *after_await,
                                    short *revents);
rund::task::Task<void> BlockedIoAwait(rund::host::io::FdView read_fd,
                                      std::atomic<std::uint32_t> *after_await,
                                      short *revents);
rund::task::Task<void> ReadyManyAwait(rund::net::SocketView socket,
                                      std::atomic<std::uint32_t> *after_await,
                                      std::uint32_t *events);
rund::task::Task<void>
ChannelSendAwait(rund::task::channel<int> *channel,
                 std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> ChannelRecvAwait(rund::task::channel<int> *channel,
                                        std::atomic<std::uint32_t> *after_await,
                                        int *value);
rund::task::Task<void> JoinAwait(rund::task::Handle child,
                                 std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> HandleAwait(std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> NestedComplete();
rund::task::Task<void> NestedTaskAwait(std::atomic<std::uint32_t> *after_await);
rund::task::Task<void> ObserveInvalidIo(std::atomic<std::uint32_t> *after_await,
                                        rund::ReasonCode *code);
rund::task::Task<void>
ObserveClosedRecv(rund::task::channel<int> *channel,
                  std::atomic<std::uint32_t> *after_await,
                  rund::ReasonCode *code);
rund::task::Task<void>
ObserveChannelCapacity(rund::task::channel<int> *channel,
                       std::atomic<std::uint32_t> *after_await,
                       rund::ReasonCode *code);
rund::task::Task<void> ObserveJoin(rund::task::Handle handle,
                                   std::atomic<std::uint32_t> *after_await,
                                   rund::ReasonCode *code);
rund::task::Task<void>
ObserveChildException(std::atomic<std::uint32_t> *after_await,
                      rund::ReasonCode *code);
int AssertCoroutineAwaitSuccess(const rund::Session::Result &report,
                                const rund::task::Status &join,
                                bool handle_valid,
                                const std::atomic<std::uint32_t> &after_await,
                                std::uint64_t expected_spawned,
                                std::uint64_t expected_completed);
int CheckCoroutineComplete();
int CheckCoroutineYield();
int CheckCoroutineSleep();
int CheckCoroutineReadyIo();
int CheckCoroutineBlockedIo();
int CheckCoroutineReadyMany();
int CheckDiscardedReadyOps();
int CheckCoroutineChannelSend();
int CheckCoroutineChannelRecv();
int CheckCoroutineChannelRaii();
int CheckDiscardedChannelOps();
int CheckCoroutineJoinAwait();
int CheckCoroutineHandleAwait();
int CheckCoroutineNestedTask();
int CheckNestedResultReuseAndLeafBoundary();
int CheckDiscardedOperations();

int RunRuntimeTaskCoroutineLifecycleContract();
int RunRuntimeTaskCoroutineFailureContract();
int RunRuntimeTaskCoroutineCancellationContract();

} // namespace rund::node::test_contract::coroutine
