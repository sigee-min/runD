#pragma once

#include <cstdint>
#include <vector>

#include "../state/model/stop.hpp"
#include "../state/storage.hpp"

namespace rund::node::scheduler_cancel {

[[nodiscard]] StopSourceRecord *
FindStopSource(std::vector<StopSourceRecord> &sources,
               std::uint64_t scheduler_id, std::uint64_t source_id,
               std::uint64_t generation, std::uint64_t epoch) noexcept;

[[nodiscard]] const StopSourceRecord *
FindStopSource(const std::vector<StopSourceRecord> &sources,
               std::uint64_t scheduler_id, std::uint64_t source_id,
               std::uint64_t generation, std::uint64_t epoch) noexcept;

} // namespace rund::node::scheduler_cancel
