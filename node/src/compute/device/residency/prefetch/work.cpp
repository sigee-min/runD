#include "local.hpp"

#include <algorithm>
#include <cstring>

namespace rund::compute::detail::residency {

void Prefetcher::work() noexcept {
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
    const bool fetch = std::any_of(
        requests_.begin(), requests_.begin() + count,
        [](const PrefetchRequest request) { return request.fetch; });
    const std::uint64_t started = fetch ? prefetch_detail::now_ns() : 0u;
    Status status = backing == nullptr ? Status::fail(Reason::PipelineInvalid)
                                       : Status::success();
    for (std::size_t index = 0u; status && index < count; ++index) {
      const PrefetchRequest request = requests_[index];
      if (!request.fetch) {
        continue;
      }
      const auto alias = std::find_if(
          aliases_.begin(), aliases_.begin() + alias_count_,
          [&request](const AliasLease &candidate) {
            return candidate && request.alias_reuse &&
                   prefetch_detail::alias_request_matches(
                       candidate, request, request.alias_owner_token);
          });
      if (request.alias_reuse && alias == aliases_.begin() + alias_count_) {
        status = Status::fail(Reason::PipelineInvalid);
        continue;
      }
      std::memset(request.frame, 0, static_cast<std::size_t>(page_bytes_));
      if (request.reuse_bytes != 0u) {
        const std::byte *const source = request.alias_reuse
                                            ? request.alias_frame
                                            : requests_[index - 1u].frame;
        if (source == nullptr ||
            (request.alias_reuse &&
             (request.alias_nonce == 0u || request.alias_generation == 0u ||
              request.alias_owner_token == 0u))) {
          status = Status::fail(Reason::PipelineInvalid);
          continue;
        }
        std::memmove(request.frame + request.reuse_target_offset,
                     source + request.reuse_source_offset, request.reuse_bytes);
      }
      if (request.read_bytes != 0u) {
        status = backing->read(
            request.read_offset,
            std::span<std::byte>{request.frame + request.read_target_offset,
                                 request.read_bytes});
      }
      if (status) {
        pages_[index].backing_bytes = request.read_bytes;
      }
    }
    const std::uint64_t elapsed =
        fetch ? prefetch_detail::now_ns() - started : 0u;
    lock.lock();
    if (state_ == State::Stop) {
      return;
    }
    status_ = status;
    io_ns_ = elapsed;
    state_ = State::Ready;
    ready_.notify_one();
    if (completion_ != nullptr) {
      completion_->publish();
    }
  }
}

} // namespace rund::compute::detail::residency
