#pragma once

#include "../device_vsm.hpp"
#include "model.hpp"
#include "operations.hpp"

#include "../../../../hash/fnv.hpp"
#include "../../../backend.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/claim.hpp"
#include "../../../pipeline/execution/prepare.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../status.hpp"
#include "../../backing.hpp"
#include "../backing.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] inline Status status_from(const rund::AccelCheck check) noexcept {
  return check.ok ? Status::success()
                  : Status::fail(
                        project_reason(check.reason, Reason::BackendFailed));
}

} // namespace rund::compute::detail::device_vsm_product_detail
