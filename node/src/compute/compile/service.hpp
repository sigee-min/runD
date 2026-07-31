#pragma once

#include <rund/compute/async.hpp>
#include <rund/compute/compile.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace rund::compute::detail {

using CompileTask = std::function<void()>;

class CompileService final {
  struct State;

public:
  using WorkerLauncher = std::function<std::thread(CompileTask)>;

  class Reservation final {
  public:
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    Reservation(Reservation &&other) noexcept;
    Reservation &operator=(Reservation &&other) noexcept;
    ~Reservation();

    [[nodiscard]] explicit operator bool() const noexcept {
      return state_ != nullptr;
    }
    [[nodiscard]] Reason reason() const noexcept { return reason_; }

  private:
    explicit Reservation(std::shared_ptr<State> state,
                         std::size_t slot) noexcept
        : state_(std::move(state)), slot_(slot), reason_(Reason::Ok) {}
    explicit Reservation(Reason reason) noexcept : reason_(reason) {}
    void cancel() noexcept;

    std::shared_ptr<State> state_{};
    std::size_t slot_{};
    Reason reason_ = Reason::AsyncCompileUnavailable;

    friend class CompileService;
  };

  explicit CompileService(Compile resources);
  CompileService(Compile resources, WorkerLauncher launcher);
  CompileService(const CompileService &) = delete;
  CompileService &operator=(const CompileService &) = delete;
  ~CompileService();

  [[nodiscard]] Reservation reserve() noexcept;
  [[nodiscard]] Status commit(Reservation &&reservation,
                              CompileTask task) noexcept;
  [[nodiscard]] Status enqueue(CompileTask task) noexcept;
  [[nodiscard]] Status enqueue(CompileFactory factory) noexcept;
  [[nodiscard]] Compile resources() const noexcept;
  void stop() noexcept;
  void close() noexcept;

private:
  static void reclaim(State &state) noexcept;
  static void cancel(const std::shared_ptr<State> &state,
                     std::size_t slot) noexcept;
  void start(const WorkerLauncher &launcher);
  static void worker_loop(const std::shared_ptr<State> &state);

  std::shared_ptr<State> state_;
  std::vector<std::thread> workers_;
};

} // namespace rund::compute::detail
