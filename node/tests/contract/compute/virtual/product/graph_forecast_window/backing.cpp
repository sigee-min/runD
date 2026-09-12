#include "internal.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace rund_node_test_virtual::product::graph_forecast_window {
void Control::reset(const bool failure) noexcept {
  std::lock_guard lock{gate};
  if (active != 0u)
    std::terminate();
  peak = 0u;
  slow_started = refill_started = refilled_while_slow = timeout = false;
  fail = failure;
}
bool Control::complete() noexcept {
  std::lock_guard lock{gate};
  return active == 0u && peak == 2u && refilled_while_slow && !timeout;
}
Backing::Backing(std::shared_ptr<Control> control, const std::size_t input)
    : control_(std::move(control)), input_(input) {
  for (std::size_t i = 0u; i < values.size(); ++i)
    values[i] = i + input + 1u;
}
std::uint64_t Backing::size_bytes() const noexcept { return sizeof(values); }
rund::compute::Status Backing::read(const std::uint64_t offset,
                                    const std::span<std::byte> bytes) noexcept {
  using namespace rund::compute;
  if (offset > sizeof(values) || bytes.size() > sizeof(values) - offset)
    return Status::fail(Reason::ShapeMismatch);
  bool fail = false;
  if (control_ != nullptr) {
    std::unique_lock lock{control_->gate};
    ++control_->active;
    control_->peak = std::max(control_->peak, control_->active);
    if (offset == 0u && input_ == 1u) {
      control_->slow_started = true;
      control_->changed.notify_all();
      if (!control_->changed.wait_for(lock, std::chrono::seconds{3}, [this] {
            return control_->refill_started;
          }))
        control_->timeout = true;
      fail = control_->fail;
    }
    if (offset == 0u && input_ == 2u) {
      if (!control_->changed.wait_for(lock, std::chrono::seconds{3}, [this] {
            return control_->slow_started;
          }))
        control_->timeout = true;
    }
    if (offset == 0u && input_ == 3u) {
      // Independent workers may enter in either order. Hold the later input
      // until the slow callback has entered, then prove real overlap/refill.
      if (!control_->changed.wait_for(lock, std::chrono::seconds{3}, [this] {
            return control_->slow_started;
          }))
        control_->timeout = true;
      control_->refill_started = true;
      control_->refilled_while_slow =
          control_->slow_started && control_->active == 2u;
      control_->changed.notify_all();
    }
    fail = fail || control_->timeout;
    --control_->active;
  }
  if (fail)
    return Status::fail(Reason::BackendFailed);
  std::memcpy(bytes.data(),
              reinterpret_cast<const std::byte *>(values.data()) + offset,
              bytes.size());
  return Status::success();
}
rund::compute::Status
Backing::write(const std::uint64_t offset,
               const std::span<const std::byte> bytes) noexcept {
  if (offset > sizeof(values) || bytes.size() > sizeof(values) - offset)
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  std::memcpy(reinterpret_cast<std::byte *>(values.data()) + offset,
              bytes.data(), bytes.size());
  return rund::compute::Status::success();
}
} // namespace rund_node_test_virtual::product::graph_forecast_window
