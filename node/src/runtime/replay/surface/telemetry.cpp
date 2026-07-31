#include "local.hpp"

#include "../../session/result.hpp"
#include "../scope/timing.hpp"

#include <node/runtime/replay/code.hpp>
#include <rund/counter.hpp>

#include <utility>

namespace rund::replay::detail::surface {

void publish(scope::Timing &timing, telemetry::Event &event,
             Session::Result *const result) noexcept {
  event.detail = timing.stop();
  if (result != nullptr) {
    ::rund::detail::session::ResultAccess::timing(*result, event.detail,
                                                  timing.reads());
  }
}

} // namespace rund::replay::detail::surface

namespace rund::replay::detail {

telemetry::Event Access::event(const telemetry::Mode mode,
                               const telemetry::Preparation plan,
                               const std::uint64_t choices,
                               const Session::Result &result) noexcept {
  const std::uint64_t input_rows =
      ::rund::detail::session::ResultAccess::input_rows(result);
  return telemetry::Event{
      .source = telemetry::Source::Replay,
      .scope = result.scope(),
      .replay =
          {
              .code = node::replay_detail::code(result.code()),
              .mode = mode,
              .plan = plan,
              .input_rows = input_rows,
              .input_bytes =
                  ::rund::detail::session::ResultAccess::input_bytes(result),
              .produced_rows = mode == telemetry::Mode::Live ||
                                       mode == telemetry::Mode::Record
                                   ? input_rows
                                   : 0u,
              .choices = choices,
              .evidence_rows = ::rund::detail::counter::SaturatingAdd(
                  ::rund::detail::counter::SaturatingAdd(
                      surface::count(result.observations().size()),
                      surface::count(result.events().size())),
                  surface::count(result.trace().records.size())),
          },
      .queue =
          {
              .depth = result.tasks().max_ready_depth(),
              .capacity =
                  ::rund::detail::session::ResultAccess::ready_capacity(result),
          },
  };
}

telemetry::Event Access::event(const telemetry::Mode mode,
                               const telemetry::Preparation plan,
                               const std::uint64_t choices,
                               const Code code) noexcept {
  return telemetry::Event{
      .source = telemetry::Source::Replay,
      .replay =
          {
              .code = code,
              .mode = mode,
              .plan = plan,
              .choices = choices,
          },
  };
}

void Access::evidence(telemetry::Event &event, const Record &record) noexcept {
  if (!record.data_) {
    return;
  }
  const node::RuntimeReplayRecord &value = record.data_->record;
  const StorageReport storage = record.storage_report();
  const DiagnosticReport diagnostic =
      value.host.payload_archive.diagnostic.report;
  const std::uint64_t host_rows = ::rund::detail::counter::SaturatingAdd(
      surface::count(value.observations.size()),
      surface::count(value.host.events.size()));
  const std::uint64_t payload_rows = ::rund::detail::counter::SaturatingAdd(
      surface::count(value.host.payload_archive.records.size()),
      surface::count(value.host.payload_archive.diagnostic.records.size()));
  event.replay.evidence_rows = ::rund::detail::counter::SaturatingAdd(
      ::rund::detail::counter::SaturatingAdd(host_rows, payload_rows),
      surface::count(value.trace.records.size()));
  event.replay.evidence_bytes = ::rund::detail::counter::SaturatingAdd(
      storage.encoded_bytes, diagnostic.retained_bytes);
  event.replay.retained_bytes = ::rund::detail::counter::SaturatingAdd(
      storage.retained_bytes, diagnostic.retained_bytes);
  event.replay.copied_bytes = ::rund::detail::counter::SaturatingAdd(
      storage.copied_bytes, diagnostic.retained_bytes);
  event.replay.physical_bytes = storage.physical_bytes;
  event.replay.allocated_bytes = storage.allocated_bytes;
  event.replay.reserved_bytes = storage.reserved_bytes;
  event.replay.storage_growths = storage.growths;
  event.replay.result_hash = record.hash();
}

void Access::emit(Session &session, telemetry::Event &&event) noexcept {
  session.emit(std::move(event));
}

Record Access::reject(Session &session, const Code code,
                      scope::Timing &timing) {
  Record output = fail_record(code);
  telemetry::Event observed = event(
      telemetry::Mode::Record, telemetry::Preparation::None, 0u, output.code());
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

Check Access::reject(Session &session, const Record &expected, const Code code,
                     const telemetry::Preparation plan, scope::Timing &timing) {
  Check output = fail_check(expected, code);
  telemetry::Event observed =
      event(telemetry::Mode::Replay, plan, 0u, output.code());
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

Scenario Access::reject(Session &session, const Code code,
                        const telemetry::Preparation plan,
                        const std::uint64_t choices, scope::Timing &timing) {
  Scenario output = fail_scenario(code);
  telemetry::Event observed =
      event(telemetry::Mode::Scenario, plan, choices, output.code());
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

} // namespace rund::replay::detail
