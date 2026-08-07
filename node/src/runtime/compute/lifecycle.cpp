#include "state.hpp"

namespace rund::node::runtime_detail {

ComputeHostAdmission ComputeHostLifecycle::admission() const noexcept {
  switch (phase_) {
  case ComputeHostPhase::Running:
    return ComputeHostAdmission::Open;
  case ComputeHostPhase::Draining:
  case ComputeHostPhase::Closing:
    return ComputeHostAdmission::Draining;
  case ComputeHostPhase::Configured:
    return ComputeHostAdmission::Standby;
  case ComputeHostPhase::Constructing:
  case ComputeHostPhase::Retiring:
  case ComputeHostPhase::Closed:
    return ComputeHostAdmission::Offline;
  }
}

bool ComputeHostLifecycle::bindable() const noexcept {
  return phase_ != ComputeHostPhase::Closing &&
         phase_ != ComputeHostPhase::Retiring &&
         phase_ != ComputeHostPhase::Closed;
}

bool ComputeHostLifecycle::closed() const noexcept {
  return phase_ == ComputeHostPhase::Closed;
}

bool ComputeHostLifecycle::configure() noexcept {
  if (phase_ != ComputeHostPhase::Constructing) {
    return false;
  }
  phase_ = ComputeHostPhase::Configured;
  return true;
}

bool ComputeHostLifecycle::start() noexcept {
  if (phase_ != ComputeHostPhase::Configured) {
    return false;
  }
  phase_ = ComputeHostPhase::Running;
  return true;
}

bool ComputeHostLifecycle::stop() noexcept {
  if (phase_ != ComputeHostPhase::Running) {
    return false;
  }
  phase_ = ComputeHostPhase::Draining;
  return true;
}

ComputeHostCloseClaim ComputeHostLifecycle::claim_close() noexcept {
  switch (phase_) {
  case ComputeHostPhase::Constructing:
  case ComputeHostPhase::Configured:
    phase_ = ComputeHostPhase::Retiring;
    return ComputeHostCloseClaim::Own;
  case ComputeHostPhase::Running:
  case ComputeHostPhase::Draining:
    phase_ = ComputeHostPhase::Closing;
    return ComputeHostCloseClaim::Own;
  case ComputeHostPhase::Closing:
  case ComputeHostPhase::Retiring:
    return ComputeHostCloseClaim::Wait;
  case ComputeHostPhase::Closed:
    return ComputeHostCloseClaim::Closed;
  }
}

void ComputeHostLifecycle::begin_retirement() noexcept {
  if (phase_ == ComputeHostPhase::Closing) {
    phase_ = ComputeHostPhase::Retiring;
  }
}

void ComputeHostLifecycle::publish_closed() noexcept {
  phase_ = ComputeHostPhase::Closed;
}

void BindLifecycle(const std::shared_ptr<ComputeHostState> &host,
                   const ComputeHostState::TaskControl cancel,
                   const ComputeHostState::TaskControl retire) noexcept {
  if (host == nullptr) {
    return;
  }
  const std::lock_guard lock{host->mutex};
  if (!host->lifecycle.bindable()) {
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
    if (!host->lifecycle.stop()) {
      return;
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
    const ComputeHostCloseClaim claim = host->lifecycle.claim_close();
    if (claim == ComputeHostCloseClaim::Closed) {
      return;
    }
    if (claim == ComputeHostCloseClaim::Wait) {
      host->drained.wait(lock, [&] { return host->lifecycle.closed(); });
      return;
    }
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
    if (host->lifecycle.closed()) {
      return;
    }
    if (host->outstanding == 0u) {
      host->lifecycle.begin_retirement();
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
        lock,
        [&] { return host->outstanding == 0u || host->lifecycle.closed(); });
    if (host->lifecycle.closed()) {
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
    host->lifecycle.publish_closed();
  }
  host->drained.notify_all();
}

} // namespace rund::node::runtime_detail
