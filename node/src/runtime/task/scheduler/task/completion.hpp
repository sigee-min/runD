#pragma once

#include "completion/model.hpp"

#include <rund/task/status.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::detail::task {
struct CoroutineResult;
struct ResultRef;
template <typename T> class ResultHandle;
} // namespace rund::detail::task

namespace rund::task {
template <typename T> class Result;
}

namespace rund::node {

class CompletionPool final {
public:
  CompletionPool() noexcept = default;
  ~CompletionPool();
  CompletionPool(const CompletionPool &) = delete;
  CompletionPool &operator=(const CompletionPool &) = delete;

  [[nodiscard]] task::Status configure(CompletionLimits limits) noexcept;
  void reset() noexcept;
  [[nodiscard]] std::uint32_t resident_cells() const noexcept;
  [[nodiscard]] CompletionLease claim() noexcept;
  [[nodiscard]] CompletionLease lease(CompletionSlot slot) const noexcept;
  [[nodiscard]] static CompletionSlot slot(CompletionLease lease) noexcept;
  [[nodiscard]] ::rund::detail::task::ResultRef
  observe_ref(CompletionLease lease) noexcept;
  template <typename T>
  [[nodiscard]] ::rund::detail::task::ResultHandle<T>
  observe(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Status transition(CompletionLease lease,
                                               task::Phase next) noexcept;
  [[nodiscard]] static task::Poll poll(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Status wait(CompletionLease lease) noexcept;
  [[nodiscard]] static bool park(CompletionLease lease,
                                 CompletionWaiter &waiter) noexcept;
  [[nodiscard]] static bool unpark(CompletionLease lease,
                                   CompletionWaiter &waiter) noexcept;
  static void release(CompletionLease lease) noexcept;

  [[nodiscard]] static task::Status complete(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Status
  prepare(CompletionLease lease,
          const ::rund::detail::task::CoroutineResult &result) noexcept;
  [[nodiscard]] static task::Status prepare_failure(CompletionLease lease,
                                                    ReasonCode code) noexcept;
  [[nodiscard]] static task::Status publish(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Status terminate(CompletionLease lease,
                                              ReasonCode code) noexcept;

  template <typename T>
    requires std::is_nothrow_move_constructible_v<T>
  [[nodiscard]] static task::Status complete(CompletionLease lease,
                                             T value) noexcept;

  template <typename T>
  [[nodiscard]] static task::Result<T> read(CompletionLease lease);

private:
  struct Store;
  using MoveFn = void (*)(void *, void *);
  using CopyFn = void (*)(void *, const void *);
  using DestroyFn = void (*)(void *) noexcept;

  template <typename T> [[nodiscard]] static const void *type_tag() noexcept;

  template <typename T> static void Move(void *out, void *value) noexcept;

  template <typename T> static void Copy(void *out, const void *value);

  template <typename T> static void Destroy(void *value) noexcept;

  static void drop(Store *store) noexcept;
  [[nodiscard]] static bool retain_observer(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Poll
  observer_poll(void *authority, std::uint32_t slot,
                std::uint32_t generation) noexcept;
  [[nodiscard]] static task::Status
  observer_wait(void *authority, std::uint32_t slot,
                std::uint32_t generation) noexcept;
  [[nodiscard]] static task::Status
  observer_copy(void *authority, std::uint32_t slot, std::uint32_t generation,
                void *out, const void *type, CopyFn copy);
  static void release_observer(void *authority, std::uint32_t slot,
                               std::uint32_t generation) noexcept;
  [[nodiscard]] static task::Status
  prepare_complete(CompletionLease lease) noexcept;
  [[nodiscard]] static task::Status prepare_raw(CompletionLease lease,
                                                void *value, std::size_t bytes,
                                                std::size_t alignment,
                                                const void *type, MoveFn move,
                                                DestroyFn destroy) noexcept;
  [[nodiscard]] static task::Status copy_raw(CompletionLease lease, void *out,
                                             const void *type, CopyFn copy);

  Store *store_{};
};

} // namespace rund::node
