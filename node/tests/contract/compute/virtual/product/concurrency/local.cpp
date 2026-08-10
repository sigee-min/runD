#include "local.hpp"

#include <chrono>
#include <cstring>

namespace rund_node_test_virtual::product::concurrency {
namespace {

using namespace std::chrono_literals;

} // namespace

void CallbackProbe::arm_first_callback() noexcept {
  std::lock_guard lock{gate_};
  block_first_ = true;
  entered_ = false;
  released_ = false;
}

void CallbackProbe::enter() noexcept {
  const std::uint32_t active = active_.fetch_add(1u) + 1u;
  std::uint32_t observed = max_active_.load();
  while (observed < active &&
         !max_active_.compare_exchange_weak(observed, active)) {
  }
  callbacks_.fetch_add(1u);

  std::unique_lock lock{gate_};
  if (!block_first_) {
    return;
  }
  block_first_ = false;
  entered_ = true;
  changed_.notify_all();
  changed_.wait(lock, [this] { return released_; });
}

void CallbackProbe::leave() noexcept { active_.fetch_sub(1u); }

bool CallbackProbe::wait_until_entered() noexcept {
  std::unique_lock lock{gate_};
  return changed_.wait_for(lock, 5s, [this] { return entered_; });
}

void CallbackProbe::release() noexcept {
  {
    std::lock_guard lock{gate_};
    released_ = true;
  }
  changed_.notify_all();
}

std::uint32_t CallbackProbe::callbacks() const noexcept {
  return callbacks_.load();
}

std::uint32_t CallbackProbe::max_active() const noexcept {
  return max_active_.load();
}

ConcurrentBacking::ConcurrentBacking(const std::size_t bytes,
                                     std::shared_ptr<CallbackProbe> probe,
                                     const std::byte value) noexcept
    : bytes_(bytes, value), probe_(std::move(probe)) {}

std::uint64_t ConcurrentBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::Status
ConcurrentBacking::read(const std::uint64_t offset,
                        const std::span<std::byte> output) noexcept {
  probe_->enter();
  const rund::compute::Status status = copy_out(offset, output);
  probe_->leave();
  return status;
}

rund::compute::Status
ConcurrentBacking::write(const std::uint64_t offset,
                         const std::span<const std::byte> input) noexcept {
  probe_->enter();
  const rund::compute::Status status = copy_in(offset, input);
  probe_->leave();
  return status;
}

bool ConcurrentBacking::contains(const std::uint64_t offset,
                                 const std::size_t bytes) const noexcept {
  return offset <= bytes_.size() && bytes <= bytes_.size() - offset;
}

rund::compute::Status
ConcurrentBacking::copy_out(const std::uint64_t offset,
                            const std::span<std::byte> output) noexcept {
  std::lock_guard lock{storage_gate_};
  if (!contains(offset, output.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  return rund::compute::Status::success();
}

rund::compute::Status
ConcurrentBacking::copy_in(const std::uint64_t offset,
                           const std::span<const std::byte> input) noexcept {
  std::lock_guard lock{storage_gate_};
  if (!contains(offset, input.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  return rund::compute::Status::success();
}

void Completion::publish(const rund::compute::Status status) noexcept {
  {
    std::lock_guard lock{gate_};
    status_ = status;
    done_ = true;
  }
  changed_.notify_all();
}

bool Completion::wait() noexcept {
  std::unique_lock lock{gate_};
  return changed_.wait_for(lock, 5s, [this] { return done_; });
}

rund::compute::Status Completion::status() const noexcept {
  std::lock_guard lock{gate_};
  return status_;
}

void finish_worker(std::thread &worker, const bool completed) noexcept {
  if (completed) {
    worker.join();
  } else {
    worker.detach();
  }
}

} // namespace rund_node_test_virtual::product::concurrency
