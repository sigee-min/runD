#include "local.hpp"

#include "../../../allocation.hpp"
#include "src/compute/compile/service.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace node_compute_cache_contract {
namespace {

inline constexpr rund::compute::Compile DefaultCompile{};
static_assert(DefaultCompile.workers == 2u);
static_assert(DefaultCompile.capacity == 64u);

[[nodiscard]] bool service_width_and_shutdown_are_bounded() {
  std::atomic<std::size_t> starts{0u};
  std::atomic<std::size_t> completed{0u};
  bool admitted = true;
  {
    rund::compute::detail::CompileService service{
        {.workers = 2u, .capacity = 8u},
        [&](rund::compute::detail::CompileTask task) {
          starts.fetch_add(1u, std::memory_order_relaxed);
          return std::thread{std::move(task)};
        }};
    for (std::size_t index = 0u; index < 8u; ++index) {
      admitted =
          admitted && service
                          .enqueue([&] {
                            completed.fetch_add(1u, std::memory_order_relaxed);
                          })
                          .ok();
    }
  }
  return admitted && starts.load(std::memory_order_relaxed) == 2u &&
         completed.load(std::memory_order_relaxed) == 8u;
}

[[nodiscard]] bool service_capacity_fifo_and_drain_are_exact() {
  std::mutex mutex;
  std::condition_variable changed;
  bool blocker_started = false;
  bool release_blocker = false;
  std::vector<std::size_t> order;
  std::size_t launches = 0u;
  bool admitted = true;
  rund::compute::Status overflow = rund::compute::Status::success();
  {
    rund::compute::detail::CompileService service{
        {.workers = 1u, .capacity = 3u},
        [&](rund::compute::detail::CompileTask task) {
          ++launches;
          return std::thread{std::move(task)};
        }};
    if (service.resources().workers != 1u ||
        service.resources().capacity != 3u) {
      return false;
    }
    admitted = service
                   .enqueue([&] {
                     std::unique_lock lock{mutex};
                     blocker_started = true;
                     changed.notify_one();
                     changed.wait(lock, [&] { return release_blocker; });
                   })
                   .ok();
    {
      std::unique_lock lock{mutex};
      changed.wait(lock, [&] { return blocker_started; });
    }
    for (std::size_t index = 0u; index < 3u; ++index) {
      admitted = admitted && service
                                 .enqueue([&, index] {
                                   const std::lock_guard lock{mutex};
                                   order.push_back(index);
                                 })
                                 .ok();
    }
    overflow = service.enqueue([] {});
    {
      const std::lock_guard lock{mutex};
      release_blocker = true;
    }
    changed.notify_one();
  }
  return admitted && !overflow &&
         overflow.reason() == rund::compute::Reason::AsyncCompileCapacity &&
         order == std::vector<std::size_t>{0u, 1u, 2u};
}

[[nodiscard]] bool service_full_rejection_is_allocation_free() {
  std::mutex mutex;
  std::condition_variable changed;
  bool blocker_started = false;
  bool release_blocker = false;
  std::size_t factory_calls = 0u;
  std::uint64_t allocations = 1u;
  bool queue_filled = false;
  rund::compute::Status overflow = rund::compute::Status::success();
  {
    rund::compute::detail::CompileService service{
        {.workers = 1u, .capacity = 1u}};
    if (!service
             .enqueue([&] {
               std::unique_lock lock{mutex};
               blocker_started = true;
               changed.notify_one();
               changed.wait(lock, [&] { return release_blocker; });
             })
             .ok()) {
      return false;
    }
    {
      std::unique_lock lock{mutex};
      changed.wait(lock, [&] { return blocker_started; });
    }
    queue_filled = service.enqueue([] {}).ok();
    const rund::compute::detail::CompileFactory factory{
        .context = std::addressof(factory_calls),
        .build =
            [](void *const raw, std::function<void()> &task) noexcept {
              ++*static_cast<std::size_t *>(raw);
              task = [] {};
              return rund::compute::Status::success();
            },
    };
    if (queue_filled) {
      node_compute_allocation::Start();
      overflow = service.enqueue(factory);
      node_compute_allocation::Stop();
      allocations = node_compute_allocation::Count();
    }
    {
      const std::lock_guard lock{mutex};
      release_blocker = true;
    }
    changed.notify_one();
  }
  return queue_filled && !overflow &&
         overflow.reason() == rund::compute::Reason::AsyncCompileCapacity &&
         factory_calls == 0u && allocations == 0u;
}

[[nodiscard]] bool service_reservations_are_fifo_and_cancel_safe() {
  std::mutex mutex;
  std::vector<std::size_t> order;
  {
    rund::compute::detail::CompileService service{
        {.workers = 1u, .capacity = 3u}};
    auto first = service.reserve();
    auto second = service.reserve();
    auto third = service.reserve();
    if (!first || !second || !third) {
      return false;
    }
    const auto task = [&](const std::size_t index) {
      return [&, index] {
        const std::lock_guard lock{mutex};
        order.push_back(index);
      };
    };
    if (!service.commit(std::move(third), task(2u)) ||
        !service.commit(std::move(second), task(1u)) ||
        !service.commit(std::move(first), task(0u))) {
      return false;
    }
    service.close();
  }
  if (order != std::vector<std::size_t>{0u, 1u, 2u}) {
    return false;
  }

  std::size_t completed = 0u;
  {
    rund::compute::detail::CompileService service{
        {.workers = 1u, .capacity = 1u}};
    {
      auto abandoned = service.reserve();
      if (!abandoned) {
        return false;
      }
    }
    auto replacement = service.reserve();
    if (!replacement ||
        !service.commit(std::move(replacement), [&] { ++completed; })) {
      return false;
    }
    service.close();
  }
  return completed == 1u;
}

[[nodiscard]] bool service_factory_failure_releases_reservation() {
  std::size_t context = 0u;
  std::size_t completed = 0u;
  rund::compute::detail::CompileService service{
      {.workers = 1u, .capacity = 1u}};
  const rund::compute::detail::CompileFactory allocating{
      .context = std::addressof(context),
      .build =
          [](void *, std::function<void()> &task) noexcept {
            try {
              auto owner = std::make_shared<std::size_t>(1u);
              task = [owner = std::move(owner)] {};
              return rund::compute::Status::success();
            } catch (const std::bad_alloc &) {
              return rund::compute::Status::fail(
                  rund::compute::Reason::AsyncCompileUnavailable);
            }
          },
  };
  node_compute_allocation::FailNext();
  const rund::compute::Status failed = service.enqueue(allocating);
  const rund::compute::Status recovered = service.enqueue([&] { ++completed; });
  service.close();
  return !failed &&
         failed.reason() == rund::compute::Reason::AsyncCompileUnavailable &&
         recovered && completed == 1u;
}

[[nodiscard]] bool service_envelopes_are_independent() {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t blockers = 0u;
  bool release = false;
  std::size_t small_completed = 0u;
  std::vector<std::size_t> large_order;
  bool admitted = true;
  rund::compute::Status small_overflow = rund::compute::Status::success();
  rund::compute::Status large_overflow = rund::compute::Status::success();
  {
    rund::compute::detail::CompileService small{
        {.workers = 1u, .capacity = 1u}};
    rund::compute::detail::CompileService large{
        {.workers = 1u, .capacity = 3u}};
    const auto blocker = [&] {
      std::unique_lock lock{mutex};
      ++blockers;
      changed.notify_all();
      changed.wait(lock, [&] { return release; });
    };
    admitted = small.enqueue(blocker).ok() && large.enqueue(blocker).ok();
    {
      std::unique_lock lock{mutex};
      changed.wait(lock, [&] { return blockers == 2u; });
    }
    admitted = admitted && small
                               .enqueue([&] {
                                 const std::lock_guard lock{mutex};
                                 ++small_completed;
                               })
                               .ok();
    for (std::size_t index = 0u; index < 3u; ++index) {
      admitted = admitted && large
                                 .enqueue([&, index] {
                                   const std::lock_guard lock{mutex};
                                   large_order.push_back(index);
                                 })
                                 .ok();
    }
    small_overflow = small.enqueue([] {});
    large_overflow = large.enqueue([] {});
    admitted = admitted && small.resources().capacity == 1u &&
               large.resources().capacity == 3u;
    {
      const std::lock_guard lock{mutex};
      release = true;
    }
    changed.notify_all();
  }
  return admitted && !small_overflow && !large_overflow &&
         small_overflow.reason() ==
             rund::compute::Reason::AsyncCompileCapacity &&
         large_overflow.reason() ==
             rund::compute::Reason::AsyncCompileCapacity &&
         small_completed == 1u &&
         large_order == std::vector<std::size_t>{0u, 1u, 2u};
}

[[nodiscard]] bool partial_start_is_transactional() {
  std::size_t starts = 0u;
  try {
    rund::compute::detail::CompileService service{
        {.workers = 2u, .capacity = 1u}, [&](auto function) -> std::thread {
          if (++starts == 2u) {
            throw std::system_error{std::make_error_code(
                std::errc::resource_unavailable_try_again)};
          }
          return std::thread{std::move(function)};
        }};
  } catch (const std::system_error &) {
    return starts == 2u;
  }
  return false;
}

} // namespace

int RunService() {
  if (!service_width_and_shutdown_are_bounded()) {
    return 1;
  }
  if (!partial_start_is_transactional()) {
    return 2;
  }
  if (!service_capacity_fifo_and_drain_are_exact()) {
    return 3;
  }
  if (!service_full_rejection_is_allocation_free()) {
    return 4;
  }
  if (!service_reservations_are_fifo_and_cancel_safe()) {
    return 5;
  }
  if (!service_factory_failure_releases_reservation()) {
    return 6;
  }
  if (!service_envelopes_are_independent()) {
    return 7;
  }
  return 0;
}

} // namespace node_compute_cache_contract
