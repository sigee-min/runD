#include "checkpoint/local.hpp"

#include "exception.hpp"

namespace rund::replay {

bool Checkpoint::ok() const noexcept { return code() == Code::Ok; }

Code Checkpoint::code() const noexcept {
  if (data_) {
    return data_->valid() ? Code::Ok : Code::CheckpointInvalid;
  }
  return code_ == Code::Ok ? Code::CheckpointMovedFrom : code_;
}

std::string_view Checkpoint::error() const noexcept {
  return ::rund::replay::error(code());
}

std::uint64_t Checkpoint::segment_count() const noexcept {
  return ok() ? data_->segment_count : 0u;
}

std::uint64_t Checkpoint::input_position() const noexcept {
  return ok() ? data_->input_position : 0u;
}

std::uint64_t Checkpoint::schema() const noexcept {
  return ok() ? data_->state_schema : 0u;
}

std::size_t Checkpoint::state_size() const noexcept {
  return ok() ? data_->state.size() : 0u;
}

std::span<const std::byte> Checkpoint::state() const noexcept {
  return ok() ? std::span<const std::byte>{data_->state}
              : std::span<const std::byte>{};
}

std::uint64_t Checkpoint::state_hash() const noexcept {
  return ok() ? data_->state_hash : 0u;
}

std::uint64_t Checkpoint::boundary_hash() const noexcept {
  return ok() ? data_->boundary_hash : 0u;
}

std::uint64_t Checkpoint::prefix_hash() const noexcept {
  return ok() ? data_->prefix_hash : 0u;
}

std::uint64_t Checkpoint::transcript_prefix_hash() const noexcept {
  return ok() ? data_->transcript_prefix_hash : 0u;
}

std::uint64_t Checkpoint::hash() const noexcept {
  return ok() ? data_->checkpoint_hash : 0u;
}

} // namespace rund::replay
