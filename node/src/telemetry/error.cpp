#include <rund/telemetry/event.hpp>

namespace rund::telemetry {

std::string_view Event::error() const noexcept {
  if (source == Source::Replay) {
    return replay::error(replay.code);
  }
  if (source != Source::Compute) {
    return "telemetry_source_invalid";
  }
  switch (compute.code) {
  case compute::Code::Ok:
    return {};
  case compute::Code::Invalid:
    return "compute_invalid";
  case compute::Code::Unsupported:
    return "compute_unsupported";
  case compute::Code::Unavailable:
    return "compute_unavailable";
  case compute::Code::Capacity:
    return "compute_capacity";
  case compute::Code::Compile:
    return "compute_compile";
  case compute::Code::Binding:
    return "compute_binding";
  case compute::Code::Transfer:
    return "compute_transfer";
  case compute::Code::Execution:
    return "compute_execution";
  }
  return "telemetry_compute_code_invalid";
}

} // namespace rund::telemetry
