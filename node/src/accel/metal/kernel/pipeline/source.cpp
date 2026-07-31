#include "source.hpp"

namespace rund::node::accel::detail {

std::string MetalPipelineSource() {
  const std::string_view status = MetalPipelineStatusSource();
  const std::string_view telemetry = MetalPipelineTelemetrySourceText();
  std::string source;
  source.reserve(status.size() + telemetry.size());
  source.append(status);
  source.append(telemetry);
  return source;
}

} // namespace rund::node::accel::detail
