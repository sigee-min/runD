#include "state.hpp"

namespace rund::node::runtime_detail {

void BindLifecycle(const std::shared_ptr<ComputeHostState> &host,
                   const ComputeHostState::TaskControl cancel,
                   const ComputeHostState::TaskControl retire) noexcept {
  if (host == nullptr) {
    return;
  }
  const std::lock_guard lock{host->mutex};
  if (host->terminalizing || host->terminal_closed) {
    return;
  }
  host->cancel_tasks = cancel;
  host->retire_tasks = retire;
}

void StopAdmission(const std::shared_ptr<ComputeHostState> &host) noexcept {
  if (host == nullptr) {
    return;
  }
  std::shared_ptr<compute::detail::CompileService> compile_service{};
  ComputeHostState::CompileControl control_compile = nullptr;
  {
    const std::lock_guard lock{host->mutex};
    if (!host->accepting) {
      return;
    }
    host->accepting = false;
    if (!host->closed && !host->terminalizing && !host->terminal_closed) {
      host->reject_reason = compute::Reason::RuntimeDraining;
    }
    compile_service = host->compile_service;
    control_compile = host->control_compile;
  }
  if (compile_service != nullptr && control_compile != nullptr) {
    control_compile(compile_service, CompileAction::Stop);
  }
}

void CloseHost(const std::shared_ptr<ComputeHostState> &host) noexcept {
  if (host == nullptr) {
    return;
  }
  StopAdmission(host);
  {
    std::unique_lock lock{host->mutex};
    if (host->terminal_closed) {
      return;
    }
    if (host->terminalizing) {
      host->drained.wait(lock, [&] { return host->terminal_closed; });
      return;
    }
    host->terminalizing = true;
  }
  std::shared_ptr<compute::detail::CompileService> compile_service{};
  ComputeHostState::CompileControl control_compile = nullptr;
  {
    std::lock_guard control{host->control};
    {
      std::lock_guard lock{host->mutex};
      compile_service = host->compile_service;
      control_compile = host->control_compile;
    }
  }
  ComputeHostState::TaskControl cancel = nullptr;
  ComputeHostState::TaskControl retire = nullptr;
  for (;;) {
    std::unique_lock lock{host->mutex};
    if (host->terminal_closed) {
      return;
    }
    host->accepting = false;
    if (host->outstanding == 0u) {
      host->closed = true;
      host->reject_reason = compute::Reason::RuntimeNotRunning;
      retire = host->retire_tasks;
      break;
    }
    cancel = host->cancel_tasks;
    lock.unlock();
    if (cancel != nullptr) {
      cancel(host);
    }
    lock.lock();
    host->drained.wait(
        lock, [&] { return host->outstanding == 0u || host->terminal_closed; });
    if (host->terminal_closed) {
      return;
    }
  }

  if (retire != nullptr) {
    retire(host);
  }
  if (compile_service != nullptr && control_compile != nullptr) {
    control_compile(compile_service, CompileAction::Close);
  }
  {
    std::lock_guard control{host->control};
    host->scheduler.Reset();
  }
  {
    std::lock_guard lock{host->mutex};
    host->signal = nullptr;
    host->signal_context = nullptr;
    host->emit = nullptr;
    host->emit_context = nullptr;
    host->cancel_tasks = nullptr;
    host->retire_tasks = nullptr;
    host->compile_service.reset();
    host->control_compile = nullptr;
    host->terminalizing = false;
    host->terminal_closed = true;
  }
  host->drained.notify_all();
}

} // namespace rund::node::runtime_detail
