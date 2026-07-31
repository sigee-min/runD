#pragma once

#include <kernel/program/compute/tile/model.hpp>

#include <functional>
#include <type_traits>

namespace rund::kernel {

template <typename Callback>
ComputeTileRunResult ComputeTileExecutor::run(Callback&& callback) {
  using CallbackObject = std::remove_reference_t<Callback>;
  static_assert(std::is_invocable_r_v<ComputeTileCallbackResult,
                                      CallbackObject&,
                                      const ComputeTile&>);
  return run_erased(
      &callback,
      [](const void* raw, const ComputeTile& tile) -> ComputeTileCallbackResult {
        auto* const typed = const_cast<CallbackObject*>(
            static_cast<const CallbackObject*>(raw));
        return std::invoke(*typed, tile);
      });
}

template <typename Callback>
ComputeTileRunResult ComputeTileExecutor::run_with(
    const WorkerBackend worker_backend, Callback&& callback) {
  using CallbackObject = std::remove_reference_t<Callback>;
  static_assert(std::is_invocable_r_v<ComputeTileCallbackResult,
                                      CallbackObject&,
                                      const ComputeTile&>);
  return run_with_erased(
      worker_backend, &callback,
      [](const void* raw, const ComputeTile& tile) -> ComputeTileCallbackResult {
        auto* const typed = const_cast<CallbackObject*>(
            static_cast<const CallbackObject*>(raw));
        return std::invoke(*typed, tile);
      });
}

} // namespace rund::kernel
