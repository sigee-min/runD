#pragma once

#include <rund/task/channel/access.hpp>
#include <rund/task/coroutine.hpp>
#include <rund/task/handle.hpp>
#include <rund/task/results.hpp>

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund::task {

template <typename T> class channel;

template <typename T> class channel {
private:
  struct Control;
  template <typename U> struct RecvDecision;

public:
  // clang-format off
#include <rund/task/channel/lifecycle.hpp>
#include <rund/task/channel/send.hpp>
#include <rund/task/channel/recv.hpp>
#include <rund/task/channel/close.hpp>
  // clang-format on

private:
  // clang-format off
#include <rund/task/channel/coroutine.hpp>
#include <rund/task/channel/status.hpp>
#include <rund/task/channel/complete.hpp>
#include <rund/task/channel/state/completion.hpp>
#include <rund/task/channel/state/gate.hpp>
#include <rund/task/channel/state/pending.hpp>
#include <rund/task/channel/state/rendezvous.hpp>
#include <rund/task/channel/state/control.hpp>
#include <rund/task/channel/waiters.hpp>
#include <rund/task/channel/waiter/queue.hpp>
#include <rund/task/channel/buffer.hpp>
#include <rund/task/channel/rendezvous/slot.hpp>
#include <rund/task/channel/send/completion.hpp>
#include <rund/task/channel/cleanup.hpp>
#include <rund/task/channel/gate.hpp>
#include <rund/task/channel/storage.hpp>
  // clang-format on
};

} // namespace rund::task
