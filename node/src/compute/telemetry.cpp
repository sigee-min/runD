#include <rund/compute/telemetry.hpp>

#include "../telemetry/compute.hpp"

namespace rund::compute::telemetry {

::rund::telemetry::Findings Profile::findings() const noexcept {
  return ::rund::telemetry::detail::ProjectEvent(
             *this, ::rund::compute::Code::Ok, ::rund::telemetry::Level::Detail)
      .findings();
}

} // namespace rund::compute::telemetry
