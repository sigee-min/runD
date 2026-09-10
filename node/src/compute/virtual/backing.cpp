#include "backing.hpp"

#include <atomic>
#include <memory>

namespace rund::compute {

namespace {
std::atomic<std::uint64_t> next_backing_id{1u};
}

VirtualBacking::VirtualBacking()
    : state_(std::make_unique<detail::VirtualBackingState>()) {
  state_->id = next_backing_id.fetch_add(1u, std::memory_order_relaxed);
  if (state_->id == 0u) {
    state_->id = next_backing_id.fetch_add(1u, std::memory_order_relaxed);
  }
}

VirtualBacking::~VirtualBacking() = default;

void VirtualBacking::publish_transaction_version() noexcept {
  ++state_->version;
  if (state_->version == 0u) {
    state_->version = 1u;
  }
}

Status VirtualBacking::invalidate() noexcept {
  if (state_ == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state_->gate};
  ++state_->version;
  if (state_->version == 0u) {
    state_->version = 1u;
  }
  return Status::success();
}

Status
VirtualBacking::read_batch(const std::span<const VirtualRead> ranges) noexcept {
  for (const VirtualRead range : ranges) {
    const Status status = read(range.offset, range.bytes);
    if (!status) {
      return status;
    }
  }
  return Status::success();
}

Status VirtualBacking::write_batch(
    const std::span<const VirtualWrite> ranges) noexcept {
  for (const VirtualWrite range : ranges) {
    const Status status = write(range.offset, range.bytes);
    if (!status) {
      return status;
    }
  }
  return Status::success();
}

} // namespace rund::compute
