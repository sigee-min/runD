#pragma once

#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <rund/task/channel.hpp>

#include <atomic>
#include <cstdint>

namespace ready_queue_detail {

rund::task::Task<void> HoldIndex(
    rund::task::channel<std::uint32_t>* gate);
rund::task::Task<void> CompleteIndex(
    std::atomic<std::uint64_t>* completed);

int CheckStorageAndOrder();
int CheckContinuations();
int CheckReuseAndBatch();
int CheckFailureOrder();

}  // namespace ready_queue_detail
