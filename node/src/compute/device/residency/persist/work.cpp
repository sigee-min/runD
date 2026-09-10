#include "../persist.hpp"

#include <chrono>

namespace rund::compute::detail::residency {
namespace {

[[nodiscard]] std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

} // namespace

void Persister::work() noexcept {
  for (;;) {
    std::unique_lock lock{gate_};
    pending_.wait(lock, [this] {
      return state_ == State::Pending || state_ == State::Stop;
    });
    if (state_ == State::Stop) {
      return;
    }
    VirtualBacking *const backing = backing_;
    const std::size_t count = count_;
    lock.unlock();

    const std::uint64_t started = now_ns();
    Status status = backing == nullptr ? Status::fail(Reason::PipelineInvalid)
                                       : Status::success();
    bool may_write = false;
    for (std::size_t index = 0u; status && index < count; ++index) {
      const PersistRequest request = requests_[index];
      may_write = true;
      status = backing->write(
          request.backing_offset,
          std::span<const std::byte>{request.frame + request.frame_offset,
                                     request.bytes});
      if (status) {
        pages_[index].bytes = request.bytes;
      }
    }
    const std::uint64_t elapsed = now_ns() - started;

    lock.lock();
    if (state_ == State::Stop) {
      return;
    }
    status_ = status;
    io_ns_ = elapsed;
    may_write_ = may_write;
    state_ = State::Ready;
    ready_.notify_one();
  }
}

} // namespace rund::compute::detail::residency
