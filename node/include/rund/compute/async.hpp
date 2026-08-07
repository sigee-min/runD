#pragma once

#include <rund/compute/flow.hpp>

#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <new>
#include <system_error>
#include <utility>

namespace rund::compute {
namespace detail {

struct CompileFactory final {
  void *context = nullptr;
  Status (*build)(void *, std::function<void()> &) noexcept = nullptr;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return context != nullptr && build != nullptr;
  }
};

[[nodiscard]] Reason
flow_reason(const std::shared_ptr<FlowState> &flow) noexcept;
[[nodiscard]] Status enqueue_compile(const std::shared_ptr<FlowState> &flow,
                                     CompileFactory factory) noexcept;

// Async compilation has one C++ exception-type projection. Each boundary
// still owns its result type and rollback (future/task reset or reservation
// cancellation); only the stable Reason selection is shared.
[[nodiscard]] inline Reason async_compile_exception_reason() noexcept {
  const std::exception_ptr error = std::current_exception();
  if (error == nullptr) {
    return Reason::ProgramCompileException;
  }
  try {
    std::rethrow_exception(error);
  } catch (const std::bad_alloc &) {
    return Reason::AsyncCompileUnavailable;
  } catch (const std::system_error &) {
    return Reason::AsyncCompileUnavailable;
  } catch (...) {
    return Reason::ProgramCompileException;
  }
}

template <class Flow> struct AsyncCompileFactory final {
  using CompileResult = decltype(std::declval<Flow &&>().compile());
  using Future = std::future<CompileResult>;

  Flow recipe;
  Future future{};

  [[nodiscard]] static Status build(void *const raw,
                                    std::function<void()> &out) noexcept {
    auto &pending = *static_cast<AsyncCompileFactory *>(raw);
    try {
      auto task = std::make_shared<std::packaged_task<CompileResult()>>(
          [recipe = std::move(pending.recipe)]() mutable {
            try {
              return std::move(recipe).compile();
            } catch (...) {
              return CompileResult::fail(async_compile_exception_reason());
            }
          });
      Future future = task->get_future();
      out = [task = std::move(task)] { (*task)(); };
      pending.future = std::move(future);
      return Status::success();
    } catch (...) {
      out = {};
      pending.future = Future{};
      return Status::fail(async_compile_exception_reason());
    }
  }
};

template <class Flow>
  requires requires(Flow &&flow) { std::move(flow).compile(); }
[[nodiscard]] auto compile_async(const std::shared_ptr<FlowState> &state,
                                 Flow flow) {
  using CompileResult = decltype(std::move(flow).compile());
  using Future = std::future<CompileResult>;
  static_assert(::rund::outcome::is_result<Result, CompileResult>,
                "compute async compilation requires a Result terminal");
  const Reason first = flow_reason(state);
  if (first != Reason::Ok) {
    return Result<Future>::fail(first);
  }
  try {
    AsyncCompileFactory<Flow> pending{.recipe = std::move(flow)};
    const Status admitted = enqueue_compile(
        state, CompileFactory{.context = std::addressof(pending),
                              .build = AsyncCompileFactory<Flow>::build});
    if (!admitted) {
      return Result<Future>::fail(admitted.reason());
    }
    if (!pending.future.valid()) {
      return Result<Future>::fail(Reason::ProgramCompileException);
    }
    return Result<Future>::success(std::move(pending.future));
  } catch (...) {
    return Result<Future>::fail(async_compile_exception_reason());
  }
}

} // namespace detail

} // namespace rund::compute
