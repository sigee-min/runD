#pragma once

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace rund_node_test_virtual::product::concurrency {

using VirtualMap = rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;

class CallbackProbe final {
public:
  void arm_first_callback() noexcept;
  void enter() noexcept;
  void leave() noexcept;
  [[nodiscard]] bool wait_until_entered() noexcept;
  void release() noexcept;
  [[nodiscard]] std::uint32_t callbacks() const noexcept;
  [[nodiscard]] std::uint32_t max_active() const noexcept;

private:
  std::atomic<std::uint32_t> active_{};
  std::atomic<std::uint32_t> max_active_{};
  std::atomic<std::uint32_t> callbacks_{};
  std::mutex gate_;
  std::condition_variable changed_;
  bool block_first_{};
  bool entered_{};
  bool released_{};
};

class ConcurrentBacking final : public rund::compute::VirtualBacking {
public:
  ConcurrentBacking(std::size_t bytes, std::shared_ptr<CallbackProbe> probe,
                    std::byte value) noexcept;

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

private:
  [[nodiscard]] bool contains(std::uint64_t offset,
                              std::size_t bytes) const noexcept;
  [[nodiscard]] rund::compute::Status
  copy_out(std::uint64_t offset, std::span<std::byte> output) noexcept;
  [[nodiscard]] rund::compute::Status
  copy_in(std::uint64_t offset, std::span<const std::byte> input) noexcept;

  std::vector<std::byte> bytes_;
  std::shared_ptr<CallbackProbe> probe_;
  std::mutex storage_gate_;
};

class Completion final {
public:
  void publish(rund::compute::Status status) noexcept;
  [[nodiscard]] bool wait() noexcept;
  [[nodiscard]] rund::compute::Status status() const noexcept;

private:
  mutable std::mutex gate_;
  std::condition_variable changed_;
  rund::compute::Status status_{
      rund::compute::Status::fail(rund::compute::Reason::PipelineInvalid)};
  bool done_{};
};

void finish_worker(std::thread &worker, bool completed) noexcept;

} // namespace rund_node_test_virtual::product::concurrency
