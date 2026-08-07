#pragma once

#include <cstdint>
#include <vector>

#include "../state/model/stop.hpp"
#include "../state/storage.hpp"

namespace rund::node::scheduler_cancel {

[[nodiscard]] StopSourceRecord *
FindStopSource(std::vector<StopSourceRecord> &sources,
               ::rund::detail::task::StopIdentity identity) noexcept;

[[nodiscard]] const StopSourceRecord *
FindStopSource(const std::vector<StopSourceRecord> &sources,
               ::rund::detail::task::StopIdentity identity) noexcept;

} // namespace rund::node::scheduler_cancel
