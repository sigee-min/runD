#pragma once

#include "../../view.hpp"

#include <cstdint>

namespace rund::compute::detail {

// One checked byte law is shared by transfer-layout planning and staging
// materialization.  The layout owner defines it; consumers only use this
// declaration so a forged element width cannot create a second interpretation.
[[nodiscard]] bool cpu_view_transfer_bytes(JobBufferView view, Type type,
                                           std::uint64_t &bytes) noexcept;

} // namespace rund::compute::detail
