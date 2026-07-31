#pragma once

#include "compute.hpp"
#include "test/assert.hpp"

#include <rund/session.hpp>

#include <cstdint>

namespace rund::node::test_contract {

struct TelemetryProbe {
  std::uint32_t events = 0u;
  ::rund::telemetry::Event event{};

  void operator()(const ::rund::telemetry::Event &next) noexcept {
    ++events;
    event = next;
  }
};

rund::SessionConfig Options(TelemetryProbe *sink = nullptr);
bool Saw(const ::rund::Trace &trace, ::rund::TraceEvent event);
} // namespace rund::node::test_contract
