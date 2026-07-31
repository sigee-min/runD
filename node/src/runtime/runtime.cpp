#include "compute/state.hpp"
#include "runtime/local.hpp"

namespace rund::telemetry::detail {

void emit(const Sink &sink, const Event &event) {
  sink.callback_(sink.context_, event);
}

} // namespace rund::telemetry::detail

namespace rund::node {

Runtime::Runtime() : state_(std::make_unique<State>()) {}

Runtime::~Runtime() { runtime_detail::CloseHost(state_->compute_host); }

void Runtime::emit(::rund::telemetry::Event &&event) noexcept {
  runtime_detail::RuntimeActiveScope active(this);
  const ::rund::telemetry::Sink sink = state_->telemetry;
  if (!sink) {
    return;
  }
  event.level = sink.level();
  event.session = state_->id;
  if (event.scope == 0u) {
    event.scope = state_->active_scope.load(std::memory_order_acquire);
  }
  if (sink.level() == ::rund::telemetry::Level::Basic) {
    event.detail = {};
  }

  try {
    ::rund::telemetry::detail::emit(sink, event);
    std::lock_guard lock{state_->mutex};
    state_->AddTrace(::rund::TraceEvent::TelemetryEmitted,
                     ::rund::TraceCode::runtime(ReasonCode::Ok));
  } catch (...) {
    std::lock_guard lock{state_->mutex};
    state_->AddTrace(
        ::rund::TraceEvent::TelemetrySkipped,
        ::rund::TraceCode::runtime(ReasonCode::TelemetrySinkFailed));
  }
}

} // namespace rund::node
