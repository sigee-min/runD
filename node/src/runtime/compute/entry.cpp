#include "request.hpp"
#include "state.hpp"
#include "local.hpp"
#include "operation.hpp"
#include "../runtime/local.hpp"
#include "../session/state.hpp"

#include "../../compute/compile/service.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace rund::compute::detail {

namespace {

void ControlCompile(const std::shared_ptr<CompileService> &service,
                    const node::runtime_detail::CompileAction action) noexcept {
  if (service == nullptr) {
    return;
  }
  if (action == node::runtime_detail::CompileAction::Close) {
    service->close();
  } else {
    service->stop();
  }
}

} // namespace

struct SessionDeviceAccess final {
  [[nodiscard]] static std::shared_ptr<node::runtime_detail::ComputeHostState>
  host(::rund::Session &session) noexcept {
    if (session.state_ == nullptr || session.state_->runtime == nullptr) {
      return {};
    }
    return node::runtime_detail::RuntimeAccess::state(*session.state_->runtime)
        .compute_host;
  }
};

Result<std::shared_ptr<DeviceState>> open_session(::rund::Session &session,
                                                  const Target target) {
  std::shared_ptr<node::runtime_detail::ComputeHostState> host =
      SessionDeviceAccess::host(session);
  if (host == nullptr) {
    return Result<std::shared_ptr<DeviceState>>::fail(Reason::RuntimeMissing);
  }
  std::lock_guard control{host->control};
  Compile resources{};
  std::uint32_t host_workers = 0u;
  std::shared_ptr<CompileService> service{};
  {
    std::lock_guard lock{host->mutex};
    if (!host->configured || host->closed || host->terminal_closed) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          Reason::RuntimeNotRunning);
    }
    if (!host->accepting) {
      return Result<std::shared_ptr<DeviceState>>::fail(host->reject_reason);
    }
    host_workers = host->workers;
    if (target.backend() == Backend::Cpu && target.workers() != 0u &&
        target.workers() != host_workers) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          Reason::NodeHostWidthMismatch);
    }
    resources = host->compile_resources;
    service = host->compile_service;
  }

  auto device = open_target(
      target.backend() == Backend::Cpu ? Target::cpu(host_workers) : target);
  if (!device) {
    return device;
  }

  bool created = false;
  if (service == nullptr) {
    try {
      service = std::make_shared<CompileService>(resources);
      created = true;
    } catch (...) {
      return Result<std::shared_ptr<DeviceState>>::fail(
          Reason::AsyncCompileUnavailable);
    }
  }

  compute::Reason rejected = Reason::Ok;
  {
    std::lock_guard lock{host->mutex};
    if (!host->configured || host->closed || host->terminal_closed) {
      rejected = Reason::RuntimeNotRunning;
    } else if (!host->accepting) {
      rejected = host->reject_reason;
    } else if (host->compile_service == nullptr) {
      host->compile_service = service;
      host->control_compile = ControlCompile;
    } else {
      service = host->compile_service;
    }
  }
  if (rejected != Reason::Ok) {
    if (created) {
      service->close();
    }
    return Result<std::shared_ptr<DeviceState>>::fail(rejected);
  }

  const Status bound = bind_compile(*device, service);
  return bound ? device
               : Result<std::shared_ptr<DeviceState>>::fail(bound.reason());
}

} // namespace rund::compute::detail

namespace rund {

compute::Request
Session::compute_job(std::shared_ptr<compute::detail::JobState> job) noexcept {
  if (state_->runtime != nullptr) {
    return state_->runtime->compute_job(std::move(job));
  }
  node::compute_detail::Operation operation =
      node::compute_detail::make_job(std::move(job));
  return compute::detail::SessionAccess::make({}, std::move(operation.owner),
                                              operation.table);
}

compute::Request Session::compute_pipeline(
    std::shared_ptr<compute::detail::PipelineState> pipeline) noexcept {
  if (state_->runtime != nullptr) {
    return state_->runtime->compute_pipeline(std::move(pipeline));
  }
  node::compute_detail::Operation operation =
      node::compute_detail::make_pipeline(std::move(pipeline));
  return compute::detail::SessionAccess::make({}, std::move(operation.owner),
                                              operation.table);
}

} // namespace rund

namespace rund::node {

::rund::compute::Request
Runtime::compute_job(std::shared_ptr<compute::detail::JobState> job) noexcept {
  compute_detail::Operation operation = compute_detail::make_job(std::move(job));
  return compute_operation(std::move(operation.owner), operation.table);
}

::rund::compute::Request Runtime::compute_pipeline(
    std::shared_ptr<compute::detail::PipelineState> pipeline) noexcept {
  compute_detail::Operation operation =
      compute_detail::make_pipeline(std::move(pipeline));
  return compute_operation(std::move(operation.owner), operation.table);
}

::rund::compute::Request
Runtime::compute_operation(std::shared_ptr<void> operation,
                           const void *const operations) noexcept {
  std::weak_ptr<runtime_detail::ComputeHostState> host{};
  {
    std::lock_guard lock{state_->mutex};
    runtime_detail::BindLifecycle(state_->compute_host,
                                  runtime_detail::CancelSubmitted,
                                  runtime_detail::RetireSubmitted);
    host = state_->compute_host;
  }
  return ::rund::compute::detail::SessionAccess::make(std::move(host),
                                                      std::move(operation),
                                                      operations);
}

} // namespace rund::node
