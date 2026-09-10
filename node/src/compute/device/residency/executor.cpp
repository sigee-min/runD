#include "executor.hpp"

#include "../../pipeline/local.hpp"

#include "../../../accel/kernel/prepared/interface/api.hpp"

#include <chrono>
#include <system_error>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

} // namespace

Executor::~Executor() {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Empty) {
      state_ = State::Stop;
    }
  }
  started_.store(true, std::memory_order_release);
  pending_.notify_one();
  ready_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool Executor::configure() noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Empty) {
    return true;
  }
  try {
    state_ = State::Idle;
    worker_ = std::thread{[this] { work(); }};
    return true;
  } catch (const std::system_error &) {
    state_ = State::Empty;
    return false;
  }
}

bool Executor::submit(const std::shared_ptr<PipelineState> &pipeline,
                      const EpochLease lease, const bool defer_generation,
                      const std::uint64_t control_generation,
                      const std::uint8_t control_parity,
                      const std::uint64_t publication_generation,
                      const std::uint8_t publication_parity) noexcept {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Idle || pipeline == nullptr || lease.token == 0u ||
        lease.bindings.empty() || pipeline->device == nullptr ||
        pipeline->device->backend != Backend::Cpu) {
      return false;
    }
    pipeline_ = pipeline;
    lease_ = lease;
    // The worker copies these through its retained submission tuple so the
    // caller may release the cursor view immediately after enqueue.
    defer_generation_ = defer_generation;
    control_generation_ = control_generation;
    control_parity_ = control_parity;
    publication_generation_ = publication_generation;
    publication_parity_ = publication_parity;
    receipt_ = {};
    started_.store(false, std::memory_order_relaxed);
    state_ = State::Pending;
  }
  pending_.notify_one();
  while (!started_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::lock_guard lock{gate_};
  return state_ == State::Running || state_ == State::Ready;
}

ExecutionReceipt Executor::wait() noexcept {
  const std::uint64_t started = now_ns();
  std::unique_lock lock{gate_};
  if (state_ == State::Pending || state_ == State::Running) {
    ready_.wait(lock, [this] {
      return state_ == State::Ready || state_ == State::Stop;
    });
  }
  if (state_ != State::Ready) {
    return ExecutionReceipt{.status = Status::fail(Reason::PipelineInvalid)};
  }
  ExecutionReceipt receipt = receipt_;
  receipt.wait_ns = now_ns() - started;
  pipeline_.reset();
  lease_ = {};
  state_ = State::Idle;
  return receipt;
}

void Executor::work() noexcept {
  for (;;) {
    std::unique_lock lock{gate_};
    pending_.wait(lock, [this] {
      return state_ == State::Pending || state_ == State::Stop;
    });
    if (state_ == State::Stop) {
      return;
    }
    const std::shared_ptr<PipelineState> pipeline = pipeline_;
    const EpochLease lease = lease_;
    const bool defer = defer_generation_;
    const std::uint64_t control_generation = control_generation_;
    const std::uint8_t control_parity = control_parity_;
    const std::uint64_t publication_generation = publication_generation_;
    const std::uint8_t publication_parity = publication_parity_;
    receipt_ = ExecutionReceipt{.started_ns = now_ns()};
    state_ = State::Running;
    started_.store(true, std::memory_order_release);
    lock.unlock();
    const Status status = run_residency_pipeline_lease(
        pipeline, lease, defer, control_generation, control_parity,
        publication_generation, publication_parity);
    const std::uint64_t completed = now_ns();
    lock.lock();
    if (state_ == State::Stop) {
      ready_.notify_one();
      return;
    }
    receipt_.status = status;
    receipt_.completed_ns = completed;
    state_ = State::Ready;
    ready_.notify_one();
  }
}

AcceleratorExecutor::~AcceleratorExecutor() {
  std::unique_lock lock{gate_};
  if (submission_.active() && state_ == State::Running) {
    ready_.wait(lock, [this] { return state_ != State::Running; });
  }
  if (state_ != State::Empty) {
    state_ = State::Stop;
  }
}

bool AcceleratorExecutor::configure() noexcept {
  std::lock_guard lock{gate_};
  if (state_ == State::Empty) {
    state_ = State::Idle;
  }
  return state_ == State::Idle;
}

bool AcceleratorExecutor::submit(
    const std::shared_ptr<PipelineState> &pipeline,
    const EpochLease lease, const bool defer_generation,
    const std::uint64_t control_generation, const std::uint8_t control_parity,
    const std::uint64_t publication_generation,
    const std::uint8_t publication_parity) noexcept {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Idle || pipeline == nullptr ||
        pipeline->device == nullptr || pipeline->device->backend == Backend::Cpu ||
        lease.token == 0u || lease.bindings.empty() || submission_.active()) {
      return false;
    }
    pipeline_ = pipeline;
    lease_ = lease;
    submission_.defer_generation = defer_generation;
    submission_.control_generation = control_generation;
    submission_.control_parity = control_parity;
    submission_.publication_generation = publication_generation;
    submission_.publication_parity = publication_parity;
    receipt_ = ExecutionReceipt{.started_ns = now_ns()};
    state_ = State::Running;
  }
  submit_residency_pipeline_lease(
      pipeline, lease, submission_, Complete, this, defer_generation,
      control_generation, control_parity, publication_generation,
      publication_parity);
  return true;
}

ExecutionReceipt AcceleratorExecutor::wait() noexcept {
  const std::uint64_t started = now_ns();
  std::unique_lock lock{gate_};
  if (state_ == State::Running) {
    ready_.wait(lock,
                [this] { return state_ == State::Ready || state_ == State::Stop; });
  }
  if (state_ != State::Ready) {
    return ExecutionReceipt{.status = Status::fail(Reason::PipelineInvalid)};
  }
  ExecutionReceipt receipt = receipt_;
  receipt.wait_ns = now_ns() - started;
  submission_.reset();
  pipeline_.reset();
  lease_ = {};
  state_ = State::Idle;
  return receipt;
}

void AcceleratorExecutor::Complete(void *const user,
                                   const Status status) noexcept {
  auto *const executor = static_cast<AcceleratorExecutor *>(user);
  if (executor != nullptr) {
    executor->complete(status);
  }
}

void AcceleratorExecutor::complete(const Status status) noexcept {
  const std::uint64_t completed = now_ns();
  std::lock_guard lock{gate_};
  if (state_ != State::Running || !submission_.active()) {
    return;
  }
  receipt_.status = status;
  receipt_.completed_ns = completed;
  state_ = State::Ready;
  ready_.notify_one();
}

AcceleratorService::~AcceleratorService() {
  {
    std::lock_guard lock{gate_};
    if (state_ != State::Empty) {
      state_ = State::Stop;
    }
  }
  pending_.notify_one();
  ready_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool AcceleratorService::configure() noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Empty) {
    return state_ == State::Idle;
  }
  try {
    state_ = State::Idle;
    worker_ = std::thread{[this] { work(); }};
    return true;
  } catch (const std::system_error &) {
    state_ = State::Empty;
    return false;
  }
}

bool AcceleratorService::submit(const AcceleratorServiceTask task,
                                void *const user) noexcept {
  {
    std::lock_guard lock{gate_};
    if ((state_ != State::Idle && state_ != State::Running) ||
        task == nullptr || user == nullptr) {
      return false;
    }
    // A backend Final may arrive after the running task returned from its
    // pump but before this worker reacquired `gate_` to publish Idle. Reuse
    // the same fixed cell as one successor; the worker observes Pending and
    // iterates without a lossy Running->Idle gap.
    task_ = task;
    user_ = user;
    state_ = State::Pending;
  }
  pending_.notify_one();
  return true;
}

void AcceleratorService::wait() noexcept {
  std::unique_lock lock{gate_};
  ready_.wait(lock, [this] {
    return state_ == State::Idle || state_ == State::Empty ||
           state_ == State::Stop;
  });
}

void AcceleratorService::work() noexcept {
  for (;;) {
    AcceleratorServiceTask task = nullptr;
    void *user = nullptr;
    {
      std::unique_lock lock{gate_};
      pending_.wait(lock, [this] {
        return state_ == State::Pending || state_ == State::Stop;
      });
      if (state_ == State::Stop) {
        return;
      }
      task = task_;
      user = user_;
      state_ = State::Running;
    }
    task(user);
    {
      std::lock_guard lock{gate_};
      if (state_ == State::Running) {
        task_ = nullptr;
        user_ = nullptr;
        state_ = State::Idle;
        ready_.notify_all();
      }
    }
  }
}

} // namespace rund::compute::detail::residency
