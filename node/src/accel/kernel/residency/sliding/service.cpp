#include "internal.hpp"

namespace rund::node::accel::detail::prepared::sliding {

State::~State() {
  if (service != nullptr) {
    service->Shutdown();
  }
}

Service::~Service() { Shutdown(); }

bool Service::Start(const Work execute, const Work terminal) noexcept {
  work = execute;
  finish = terminal;
  try {
    const std::shared_ptr<Service> retained = shared_from_this();
    worker = std::thread{[retained] {
      for (;;) {
        std::shared_ptr<State> state{};
        {
          std::unique_lock lock{retained->gate};
          retained->ready.wait(
              lock, [&] { return retained->stop || retained->pending; });
          if (retained->stop) {
            return;
          }
          state = retained->pending;
          retained->working = true;
          retained->again = false;
        }
        for (;;) {
          retained->work(state);
          {
            std::lock_guard lock{retained->gate};
            if (retained->again) {
              retained->again = false;
              continue;
            }
            retained->working = false;
            retained->pending.reset();
            state->service_queued.store(false, std::memory_order_release);
          }
          retained->finish(state);
          break;
        }
      }
    }};
  } catch (...) {
    return false;
  }
  return true;
}

void Service::Schedule(const std::shared_ptr<State> &state) noexcept {
  {
    std::lock_guard lock{gate};
    if (stop) {
      return;
    }
    state->service_queued.store(true, std::memory_order_release);
    if (working || pending != nullptr) {
      again = true;
      return;
    }
    pending = state;
  }
  ready.notify_one();
}

void Service::Shutdown() noexcept {
  {
    std::lock_guard lock{gate};
    stop = true;
  }
  ready.notify_all();
  if (!worker.joinable()) {
    return;
  }
  if (worker.get_id() == std::this_thread::get_id()) {
    worker.detach();
  } else {
    worker.join();
  }
}

void service_pump(const std::shared_ptr<State> &state) noexcept { pump(state); }

void service_finish(const std::shared_ptr<State> &state) noexcept {
  emit_final(state);
}

void schedule_service(State &state) noexcept {
  std::shared_ptr<State> retained{};
  {
    std::lock_guard lock{state.gate};
    retained = state.active_owner;
  }
  if (retained != nullptr && state.service != nullptr) {
    state.service->Schedule(retained);
  }
}

} // namespace rund::node::accel::detail::prepared::sliding
