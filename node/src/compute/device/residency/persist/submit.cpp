#include "../persist.hpp"

namespace rund::compute::detail::residency {

bool Persister::submit(VirtualBacking &backing,
                       const std::span<const PersistRequest> requests,
                       const std::uint64_t token) noexcept {
  std::lock_guard lock{gate_};
  if (state_ != State::Idle || requests.empty() ||
      requests.size() > requests_.size() || token == 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const PersistRequest request = requests[index];
    if (request.key.backing == 0u || request.frame == nullptr ||
        request.bytes == 0u || request.frame_offset > frame_bytes_ ||
        request.bytes > frame_bytes_ - request.frame_offset) {
      return false;
    }
    requests_[index] = request;
    pages_[index] = PersistedPage{
        .key = request.key,
        .backing_offset = request.backing_offset,
        .bytes = 0u,
        .physical_frame = request.physical_frame,
    };
  }
  backing_ = &backing;
  count_ = requests.size();
  token_ = token;
  status_ = Status::success();
  io_ns_ = 0u;
  may_write_ = false;
  state_ = State::Pending;
  pending_.notify_one();
  return true;
}

} // namespace rund::compute::detail::residency
