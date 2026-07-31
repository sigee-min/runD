#include "service.hpp"

#include "../device/state.hpp"
#include "../flow/state.hpp"

#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] CompileService::WorkerLauncher default_launcher() {
  return [](CompileTask task) { return std::thread{std::move(task)}; };
}

enum class SlotState : unsigned char {
  Empty,
  Reserved,
  Ready,
  Canceled,
};

struct CompileSlot final {
  std::optional<CompileTask> task{};
  SlotState state = SlotState::Empty;
};

} // namespace

struct CompileService::State final {
  explicit State(const Compile configured)
      : resources(configured), queue(configured.capacity) {}

  Compile resources{};
  std::mutex mutex{};
  std::condition_variable ready{};
  std::vector<CompileSlot> queue{};
  std::size_t head{};
  std::size_t occupied{};
  bool stopping{};
};

CompileService::Reservation::Reservation(Reservation &&other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_),
      reason_(other.reason_) {
  other.slot_ = 0u;
  other.reason_ = Reason::AsyncCompileUnavailable;
}

CompileService::Reservation &
CompileService::Reservation::operator=(Reservation &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  cancel();
  state_ = std::move(other.state_);
  slot_ = other.slot_;
  reason_ = other.reason_;
  other.slot_ = 0u;
  other.reason_ = Reason::AsyncCompileUnavailable;
  return *this;
}

CompileService::Reservation::~Reservation() { cancel(); }

void CompileService::Reservation::cancel() noexcept {
  if (state_ == nullptr) {
    return;
  }
  CompileService::cancel(state_, slot_);
  state_.reset();
  slot_ = 0u;
  reason_ = Reason::AsyncCompileUnavailable;
}

CompileService::CompileService(const Compile resources)
    : CompileService(resources, default_launcher()) {}

CompileService::CompileService(const Compile resources,
                               WorkerLauncher launcher) {
  if (resources.workers == 0u || resources.capacity == 0u) {
    throw std::invalid_argument{"compute compile resources are invalid"};
  }
  state_ = std::make_shared<State>(resources);
  start(launcher);
}

CompileService::~CompileService() { close(); }

CompileService::Reservation CompileService::reserve() noexcept {
  const std::shared_ptr<State> state = state_;
  if (state == nullptr) {
    return Reservation{Reason::AsyncCompileUnavailable};
  }
  const std::lock_guard lock{state->mutex};
  if (state->stopping) {
    return Reservation{Reason::AsyncCompileUnavailable};
  }
  if (state->occupied == state->resources.capacity) {
    return Reservation{Reason::AsyncCompileCapacity};
  }
  std::size_t tail = state->head + state->occupied;
  if (tail >= state->resources.capacity) {
    tail -= state->resources.capacity;
  }
  CompileSlot &slot = state->queue[tail];
  if (slot.state != SlotState::Empty || slot.task.has_value()) {
    return Reservation{Reason::AsyncCompileUnavailable};
  }
  slot.state = SlotState::Reserved;
  ++state->occupied;
  return Reservation{state, tail};
}

Status CompileService::commit(Reservation &&reservation,
                              CompileTask task) noexcept {
  static_assert(std::is_nothrow_move_constructible_v<CompileTask>);
  const std::shared_ptr<State> state = state_;
  if (state == nullptr || reservation.state_ != state) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
  Status result = Status::success();
  {
    const std::lock_guard lock{state->mutex};
    CompileSlot &slot = state->queue[reservation.slot_];
    if (slot.state != SlotState::Reserved || slot.task.has_value()) {
      result = Status::fail(Reason::AsyncCompileUnavailable);
    } else if (state->stopping || !task) {
      slot.state = SlotState::Canceled;
      reclaim(*state);
      result = Status::fail(Reason::AsyncCompileUnavailable);
    } else {
      slot.task.emplace(std::move(task));
      slot.state = SlotState::Ready;
    }
  }
  reservation.state_.reset();
  reservation.slot_ = 0u;
  reservation.reason_ = result.reason();
  state->ready.notify_all();
  return result;
}

Status CompileService::enqueue(CompileTask task) noexcept {
  Reservation reservation = reserve();
  if (!reservation) {
    return Status::fail(reservation.reason());
  }
  return commit(std::move(reservation), std::move(task));
}

Status CompileService::enqueue(const CompileFactory factory) noexcept {
  if (!factory.valid()) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
  Reservation reservation = reserve();
  if (!reservation) {
    return Status::fail(reservation.reason());
  }
  CompileTask task{};
  const Status built = factory.build(factory.context, task);
  if (!built) {
    return built;
  }
  return commit(std::move(reservation), std::move(task));
}

void CompileService::reclaim(State &state) noexcept {
  while (state.occupied != 0u &&
         state.queue[state.head].state == SlotState::Canceled) {
    CompileSlot &slot = state.queue[state.head];
    slot.task.reset();
    slot.state = SlotState::Empty;
    ++state.head;
    if (state.head == state.resources.capacity) {
      state.head = 0u;
    }
    --state.occupied;
  }
}

void CompileService::cancel(const std::shared_ptr<State> &state,
                            const std::size_t slot_index) noexcept {
  {
    const std::lock_guard lock{state->mutex};
    if (slot_index >= state->queue.size()) {
      return;
    }
    CompileSlot &slot = state->queue[slot_index];
    if (slot.state != SlotState::Reserved) {
      return;
    }
    slot.state = SlotState::Canceled;
    reclaim(*state);
  }
  state->ready.notify_all();
}

Compile CompileService::resources() const noexcept {
  return state_ == nullptr ? Compile{.workers = 0u, .capacity = 0u}
                           : state_->resources;
}

void CompileService::start(const WorkerLauncher &launcher) {
  try {
    workers_.reserve(state_->resources.workers);
    for (std::uint32_t index = 0u; index < state_->resources.workers; ++index) {
      const std::shared_ptr<State> state = state_;
      workers_.push_back(
          launcher([state] { CompileService::worker_loop(state); }));
      if (!workers_.back().joinable()) {
        throw std::system_error{
            std::make_error_code(std::errc::resource_unavailable_try_again)};
      }
    }
  } catch (...) {
    close();
    throw;
  }
}

void CompileService::close() noexcept {
  const std::shared_ptr<State> state = state_;
  if (state == nullptr) {
    return;
  }
  stop();
  const std::thread::id current = std::this_thread::get_id();
  for (std::thread &worker : workers_) {
    if (!worker.joinable()) {
      continue;
    }
    if (worker.get_id() == current) {
      worker.detach();
    } else {
      worker.join();
    }
  }
  workers_.clear();
}

void CompileService::stop() noexcept {
  const std::shared_ptr<State> state = state_;
  if (state == nullptr) {
    return;
  }
  {
    const std::lock_guard lock{state->mutex};
    state->stopping = true;
    for (CompileSlot &slot : state->queue) {
      if (slot.state == SlotState::Reserved) {
        slot.state = SlotState::Canceled;
      }
    }
    reclaim(*state);
  }
  state->ready.notify_all();
}

void CompileService::worker_loop(const std::shared_ptr<State> &state) {
  for (;;) {
    CompileTask task;
    {
      std::unique_lock lock{state->mutex};
      state->ready.wait(lock, [&] {
        return state->occupied == 0u
                   ? state->stopping
                   : state->queue[state->head].state != SlotState::Reserved;
      });
      reclaim(*state);
      if (state->occupied == 0u) {
        if (!state->stopping) {
          continue;
        }
        return;
      }
      CompileSlot &slot = state->queue[state->head];
      if (slot.state != SlotState::Ready || !slot.task.has_value()) {
        continue;
      }
      task = std::move(*slot.task);
      slot.task.reset();
      slot.state = SlotState::Empty;
      ++state->head;
      if (state->head == state->resources.capacity) {
        state->head = 0u;
      }
      --state->occupied;
    }
    try {
      task();
    } catch (...) {
    }
  }
}

Reason flow_reason(const std::shared_ptr<FlowState> &flow) noexcept {
  return flow == nullptr ? Reason::AsyncCompileUnavailable
                         : flow->status.reason();
}

Status enqueue_compile(const std::shared_ptr<FlowState> &flow,
                       const CompileFactory factory) noexcept {
  if (flow == nullptr || flow->device == nullptr) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
  const std::shared_ptr<CompileService> service = flow->device->compile.lock();
  return service == nullptr ? Status::fail(Reason::AsyncCompileUnavailable)
                            : service->enqueue(factory);
}

Status own_compile(const std::shared_ptr<DeviceState> &device,
                   const Compile resources) {
  if (device == nullptr || resources.workers == 0u ||
      resources.capacity == 0u) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
  try {
    auto service = std::make_shared<CompileService>(resources);
    device->compile_resources = resources;
    device->compile = service;
    device->compile_owner = std::move(service);
    return Status::success();
  } catch (...) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
}

Status bind_compile(const std::shared_ptr<DeviceState> &device,
                    const std::shared_ptr<CompileService> &service) noexcept {
  if (device == nullptr || service == nullptr) {
    return Status::fail(Reason::AsyncCompileUnavailable);
  }
  device->compile_resources = service->resources();
  device->compile = service;
  device->compile_owner.reset();
  return Status::success();
}

Compile device_compile(const std::shared_ptr<DeviceState> &state) noexcept {
  return state == nullptr ? Compile{.workers = 0u, .capacity = 0u}
                          : state->compile_resources;
}

} // namespace rund::compute::detail
