#pragma once

#include "../../pipeline/residency/submission.hpp"
#include "registry/credentials/epoch.hpp"

#include <rund/compute/status.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace rund::compute::detail {
struct PipelineState;

namespace residency {

struct ExecutionReceipt final {
  Status status{Status::success()};
  std::uint64_t started_ns{};
  std::uint64_t completed_ns{};
  std::uint64_t wait_ns{};
};

// The CPU-only cold worker. Accelerator callback state is deliberately absent
// so a CPU Pool neither retains nor branches through native async machinery.
class Executor final {
public:
  Executor() = default;
  ~Executor();
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  [[nodiscard]] bool configure() noexcept;
  [[nodiscard]] bool submit(const std::shared_ptr<PipelineState> &pipeline,
                            EpochLease lease, bool defer_generation = false,
                            std::uint64_t control_generation =
                                std::numeric_limits<std::uint64_t>::max(),
                            std::uint8_t control_parity = 0u,
                            std::uint64_t publication_generation =
                                std::numeric_limits<std::uint64_t>::max(),
                            std::uint8_t publication_parity = 0u) noexcept;
  [[nodiscard]] ExecutionReceipt wait() noexcept;

private:
  enum class State : std::uint8_t {
    Empty,
    Idle,
    Pending,
    Running,
    Ready,
    Stop
  };

  void work() noexcept;
  std::mutex gate_;
  std::condition_variable ready_;
  std::condition_variable pending_;
  std::thread worker_;
  std::shared_ptr<PipelineState> pipeline_;
  EpochLease lease_{};
  bool defer_generation_{};
  std::uint64_t control_generation_{std::numeric_limits<std::uint64_t>::max()};
  std::uint8_t control_parity_{};
  std::uint64_t publication_generation_{
      std::numeric_limits<std::uint64_t>::max()};
  std::uint8_t publication_parity_{};
  ExecutionReceipt receipt_{};
  std::atomic<bool> started_{false};
  State state_{State::Empty};
};

// One fixed accelerator-native completion receipt. It owns no OS worker and
// performs no scheduling policy: the Authority has already frozen the exact
// lease/local selection before submit().
class AcceleratorExecutor final {
public:
  AcceleratorExecutor() = default;
  ~AcceleratorExecutor();
  AcceleratorExecutor(const AcceleratorExecutor &) = delete;
  AcceleratorExecutor &operator=(const AcceleratorExecutor &) = delete;

  [[nodiscard]] bool configure() noexcept;
  [[nodiscard]] bool submit(const std::shared_ptr<PipelineState> &pipeline,
                            EpochLease lease, bool defer_generation = false,
                            std::uint64_t control_generation =
                                std::numeric_limits<std::uint64_t>::max(),
                            std::uint8_t control_parity = 0u,
                            std::uint64_t publication_generation =
                                std::numeric_limits<std::uint64_t>::max(),
                            std::uint8_t publication_parity = 0u) noexcept;
  [[nodiscard]] ExecutionReceipt wait() noexcept;

private:
  friend struct AcceleratorExecutionRing;
  enum class State : std::uint8_t { Empty, Idle, Running, Ready, Stop };

  static void Complete(void *user, Status status) noexcept;
  void complete(Status status) noexcept;

  std::mutex gate_;
  std::condition_variable ready_;
  std::shared_ptr<PipelineState> pipeline_;
  EpochLease lease_{};
  ExecutionReceipt receipt_{};
  ResidencyPipelineSubmission submission_{};
  State state_{State::Empty};
};

using AcceleratorServiceTask = void (*)(void *) noexcept;

// Accelerator-only cold serial lane for recurrent controller continuations.
// It is allocated with AcceleratorExecutionRing and therefore adds no CPU Pool
// object or hot-path footprint. One Authority permits at most one live stream,
// so a single fixed task cell is sufficient and allocation-free on warm runs.
class AcceleratorService final {
public:
  AcceleratorService() = default;
  ~AcceleratorService();
  AcceleratorService(const AcceleratorService &) = delete;
  AcceleratorService &operator=(const AcceleratorService &) = delete;

  [[nodiscard]] bool configure() noexcept;
  [[nodiscard]] bool submit(AcceleratorServiceTask, void *) noexcept;
  void wait() noexcept;

private:
  enum class State : std::uint8_t { Empty, Idle, Pending, Running, Stop };

  void work() noexcept;

  std::mutex gate_;
  std::condition_variable ready_;
  std::condition_variable pending_;
  std::thread worker_;
  AcceleratorServiceTask task_{};
  void *user_{};
  State state_{State::Empty};
};

struct AcceleratorExecutionRing final {
  std::array<AcceleratorExecutor, 2u> banks;
  AcceleratorService service;
};

} // namespace residency
} // namespace rund::compute::detail
