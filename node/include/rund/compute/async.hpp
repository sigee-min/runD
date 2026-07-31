#pragma once

#include <rund/compute/flow.hpp>

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
            } catch (const std::bad_alloc &) {
              return CompileResult::fail(Reason::AsyncCompileUnavailable);
            } catch (const std::system_error &) {
              return CompileResult::fail(Reason::AsyncCompileUnavailable);
            } catch (...) {
              return CompileResult::fail(Reason::ProgramCompileException);
            }
          });
      Future future = task->get_future();
      out = [task = std::move(task)] { (*task)(); };
      pending.future = std::move(future);
      return Status::success();
    } catch (const std::bad_alloc &) {
      out = {};
      pending.future = Future{};
      return Status::fail(Reason::AsyncCompileUnavailable);
    } catch (const std::system_error &) {
      out = {};
      pending.future = Future{};
      return Status::fail(Reason::AsyncCompileUnavailable);
    } catch (...) {
      out = {};
      pending.future = Future{};
      return Status::fail(Reason::ProgramCompileException);
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
  } catch (const std::bad_alloc &) {
    return Result<Future>::fail(Reason::AsyncCompileUnavailable);
  } catch (const std::system_error &) {
    return Result<Future>::fail(Reason::AsyncCompileUnavailable);
  } catch (...) {
    return Result<Future>::fail(Reason::ProgramCompileException);
  }
}

} // namespace detail

} // namespace rund::compute
