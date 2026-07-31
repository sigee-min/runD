#include "support.hpp"

namespace rund::node::test_contract {

rund::SessionConfig Options(TelemetryProbe *const sink) {
  rund::SessionConfig options{};
  options.id = 1u;
  options.workers = 2u;
  options.trace_capacity = 256u;
  if (sink != nullptr) {
    options.telemetry = rund::telemetry::bind(*sink);
  }
  return options;
}

bool Saw(const ::rund::Trace &trace, const ::rund::TraceEvent event) {
  for (const ::rund::TraceRecord &record : trace.records) {
    if (record.event == event) {
      return true;
    }
  }
  return false;
}

} // namespace rund::node::test_contract
