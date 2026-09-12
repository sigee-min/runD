#include "src/compute/device/residency/prefetch.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
using namespace rund::compute;
using namespace rund::compute::detail::residency;
class GatedBacking final : public VirtualBacking {
public:
  std::uint64_t size_bytes() const noexcept override { return 8u; }
  Status read(std::uint64_t,
              const std::span<std::byte> bytes) noexcept override {
    std::unique_lock lock{gate_};
    entered_ = true;
    changed_.notify_all();
    changed_.wait(lock, [this] { return released_; });
    if (fail_)
      return Status::fail(Reason::BackendFailed);
    std::memset(bytes.data(), 0x5a, bytes.size());
    return Status::success();
  }
  Status write(std::uint64_t, std::span<const std::byte>) noexcept override {
    return Status::fail(Reason::PipelineInvalid);
  }
  bool entered() noexcept {
    std::unique_lock lock{gate_};
    return changed_.wait_for(lock, std::chrono::seconds{5},
                             [this] { return entered_; });
  }
  void release() noexcept {
    std::lock_guard lock{gate_};
    released_ = true;
    changed_.notify_all();
  }
  void failure(const bool fail) noexcept {
    std::lock_guard lock{gate_};
    fail_ = fail;
  }

private:
  std::mutex gate_;
  std::condition_variable changed_;
  bool entered_{};
  bool released_{};
  bool fail_{};
};
// A broken notification must fail the contract, not hang the test process.
class Deadline final {
public:
  Deadline(PrefetchCompletion &completion, GatedBacking &a, GatedBacking &b)
      : thread_([this, &completion, &a, &b] {
          std::unique_lock lock{gate_};
          if (changed_.wait_for(lock, std::chrono::seconds{10},
                                [this] { return done_; }))
            return;
          expired_ = true;
          a.release();
          b.release();
          completion.publish();
        }) {}
  ~Deadline() {
    {
      std::lock_guard lock{gate_};
      done_ = true;
    }
    changed_.notify_all();
    thread_.join();
  }
  bool expired() noexcept {
    std::lock_guard lock{gate_};
    return expired_;
  }

private:
  std::mutex gate_;
  std::condition_variable changed_;
  bool done_{};
  bool expired_{};
  std::thread thread_;
};
} // namespace

namespace rund_node_test_pipeline_residency {
int CheckPrefetchCompletion() {
  PrefetchCompletion completion;
  std::array<Prefetcher, 2u> workers;
  GatedBacking first;
  GatedBacking second;
  Deadline deadline{completion, first, second};
  std::array<std::array<std::byte, 8u>, 2u> frames{};
  const auto request = [&](const std::size_t lane) {
    return PrefetchRequest{
        .key = {.backing = 100u + lane, .version = 1u, .page = 0u},
        .bytes = 8u,
        .read_bytes = 8u,
        .frame = frames[lane].data(),
        .physical_frame = static_cast<std::uint32_t>(lane),
        .fetch = true};
  };
  const auto a = request(0u);
  const auto b = request(1u);
  const auto finish = [&](const int result) {
    first.release();
    second.release();
    (void)workers[0u].wait();
    (void)workers[1u].wait();
    return result;
  };
  for (auto &worker : workers)
    if (!worker.configure(8u, 1u, &completion))
      return finish(1);
  const std::uint64_t initial = completion.observe();
  if (!workers[0u].submit(first, {&a, 1u}, 1u, false) ||
      !workers[1u].submit(second, {&b, 1u}, 2u, false) || !first.entered() ||
      !second.entered() || workers[0u].ready() || workers[1u].ready())
    return finish(2);
  second.release();
  completion.wait(initial);
  if (deadline.expired() || workers[0u].ready() || !workers[1u].ready())
    return finish(3);
  const auto second_receipt = workers[1u].wait();
  if (!second_receipt.status || second_receipt.token != 2u ||
      frames[1u][0u] != std::byte{0x5a})
    return finish(4);
  const auto before_first = completion.observe();
  first.release();
  const auto first_receipt = workers[0u].wait();
  // Completion already happened before wait-any: its old snapshot must not
  // park.
  completion.wait(before_first);
  if (deadline.expired() || !first_receipt.status || first_receipt.token != 1u)
    return finish(5);
  for (std::uint64_t turn = 0u; turn < 128u; ++turn) {
    const bool fail = (turn % 3u) == 0u;
    second.failure(fail);
    const auto observed = completion.observe();
    if (!workers[1u].submit(second, {&b, 1u}, turn + 3u, false))
      return finish(6);
    completion.wait(observed);
    if (deadline.expired() || !workers[1u].ready())
      return finish(7);
    const auto receipt = workers[1u].wait();
    if (receipt.token != turn + 3u ||
        static_cast<bool>(receipt.status) == fail ||
        (fail && receipt.status.reason() != Reason::BackendFailed) ||
        !workers[1u].quiescent())
      return finish(8);
  }
  PrefetchCompletion other;
  if (workers[0u].configure(8u, 1u, &other) ||
      !workers[0u].configure(8u, 1u, &completion) || !workers[0u].quiescent() ||
      !workers[1u].quiescent())
    return finish(9);
  return finish(0);
}
} // namespace rund_node_test_pipeline_residency
