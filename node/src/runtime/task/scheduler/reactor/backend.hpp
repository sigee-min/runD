#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "model.hpp"

namespace rund::node {

struct ReactorApplyResult {
  bool ok = true;
  bool invalid = false;
  bool unavailable = false;
  ReactorHandle invalid_handle = kInvalidReactorHandle;
};

[[nodiscard]] ReactorApplyResult
ReactorBackendApplyChanges(ReactorRuntime &reactor,
                           ::rund::detail::task::StatStorage &stats) noexcept;

[[nodiscard]] ReactorPlatformPollResult
ReactorBackendPoll(ReactorRuntime &reactor,
                   ::rund::detail::task::StatStorage &stats, int timeout_ms,
                   std::size_t max_events) noexcept;

void ReactorCloseRuntime(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
