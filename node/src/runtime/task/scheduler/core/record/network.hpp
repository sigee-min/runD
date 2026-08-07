#pragma once

#include <rund/host/event.hpp>
#include <rund/task/stats/storage.hpp>

namespace rund::node::record_detail {

void RecordNetworkStats(::rund::detail::task::StatStorage &stats,
                        const ::rund::host::Event &event) noexcept;

} // namespace rund::node::record_detail
