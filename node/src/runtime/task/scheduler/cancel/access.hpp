#pragma once

#include <rund/task/cancel.hpp>

#include <cstdint>

namespace rund::detail::task {

class StopTokenSourceAccess;

class StopAccess final {
public:
  StopAccess() = delete;

  [[nodiscard]] static bool Identity(::rund::task::stop_token token,
                                     std::uint64_t *scheduler_id,
                                     std::uint64_t *source_id,
                                     std::uint64_t *generation,
                                     std::uint64_t *epoch) noexcept {
    if (scheduler_id == nullptr || source_id == nullptr ||
        generation == nullptr || epoch == nullptr) {
      return false;
    }
    *scheduler_id = token.scheduler_id_;
    *source_id = token.source_id_;
    *generation = token.generation_;
    *epoch = token.epoch_;
    return token.scheduler_id_ != 0u && token.source_id_ != 0u &&
           token.generation_ != 0u && token.epoch_ != 0u;
  }
};

} // namespace rund::detail::task
