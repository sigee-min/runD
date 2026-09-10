#pragma once

#include "../../../../hash/fnv.hpp"
#include "../../../backend.hpp"
#include "../../../device/residency/execution/sliding.hpp"
#include "../../../device/residency/registry/sliding_owner.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/execution/attempt.hpp"
#include "../../../pipeline/execution/prepare.hpp"
#include "../../../pipeline/execution/submit.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../stats.hpp"
#include "../../../status.hpp"
#include "../../backing.hpp"
#include "../../state.hpp"
#include "../backing.hpp"
#include "../cache.hpp"
#include "../sliding.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>

#include "model.hpp"
#include "operations.hpp"

namespace rund::compute::detail::sliding_product_detail {

void record_service_fault(SlidingProductRun &, ServiceFaultStage, std::uint32_t,
                          std::uint64_t, Status, std::uint64_t) noexcept;
[[nodiscard]] ServiceFault snapshot_service_fault(SlidingProductRun &) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
