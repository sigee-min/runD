#include <rund/compute/telemetry.hpp>

#include "../telemetry/compute.hpp"

namespace rund::compute::telemetry {

::rund::telemetry::Findings Profile::findings() const noexcept {
  return ::rund::telemetry::detail::ComputeFindings(execution_);
}

} // namespace rund::compute::telemetry
