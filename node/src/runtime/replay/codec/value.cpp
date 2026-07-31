#include "local.hpp"

#include <rund/session/trace.hpp>
#include <rund/task/observation.hpp>

#include <bit>
#include <cstdint>
#include <limits>

namespace rund::node::replay_detail::artifact {

bool write_observations(
    Writer &out,
    const std::span<const task::Observation> observations) noexcept {
  if (!out.ok()) {
    return false;
  }
  if (!out.varuint(static_cast<std::uint64_t>(observations.size()))) {
    return false;
  }
  std::uint64_t previous_sequence = 0u;
  std::uint64_t previous_deadline = 0u;
  for (const task::Observation &observation : observations) {
    const std::uint64_t deadline =
        std::bit_cast<std::uint64_t>(observation.deadline_ns);
    std::uint64_t sequence_delta = 0u;
    std::uint64_t deadline_delta = 0u;
    if (observation.kind > task::ObservationKind::IoPollFailed ||
        !::rund::ValidReasonCode(observation.reason_code) ||
        !rund::kernel::checked::sub(observation.sequence, previous_sequence,
                                    sequence_delta) ||
        !rund::kernel::checked::sub(deadline, previous_deadline,
                                    deadline_delta)) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    if (!out.varuint(sequence_delta) ||
        !out.varuint(static_cast<std::uint64_t>(observation.kind)) ||
        !out.varuint(observation.task_id) ||
        !out.varuint(observation.wait_id) || !out.sint(observation.fd) ||
        !out.varuint(static_cast<std::uint16_t>(observation.interest)) ||
        !out.varuint(static_cast<std::uint16_t>(observation.revents)) ||
        !out.varuint(deadline_delta) ||
        !out.varuint(static_cast<std::uint64_t>(observation.reason_code))) {
      return false;
    }
    previous_sequence = observation.sequence;
    previous_deadline = deadline;
  }
  return true;
}

bool read_observations(Reader &in, Admission &admission,
                       std::vector<task::Observation> &out) {
  std::uint64_t count = 0u;
  std::size_t count_size = 0u;
  if (!in.varuint(count) || !admission.entries(count) ||
      !size(count, count_size)) {
    return false;
  }
  out.resize(count_size);
  std::uint64_t previous_sequence = 0u;
  std::uint64_t previous_deadline = 0u;
  for (task::Observation &observation : out) {
    std::uint64_t sequence_delta = 0u;
    std::uint64_t kind = 0u;
    std::int64_t fd = 0;
    std::uint64_t interest = 0u;
    std::uint64_t revents = 0u;
    std::uint64_t deadline_delta = 0u;
    std::uint64_t reason = 0u;
    if (!in.varuint(sequence_delta) || !in.varuint(kind) ||
        kind >
            static_cast<std::uint64_t>(task::ObservationKind::IoPollFailed) ||
        !in.varuint(observation.task_id) || !in.varuint(observation.wait_id) ||
        !in.sint(fd) || fd < std::numeric_limits<int>::min() ||
        fd > std::numeric_limits<int>::max() || !in.varuint(interest) ||
        interest > std::numeric_limits<std::uint16_t>::max() ||
        !in.varuint(revents) ||
        revents > std::numeric_limits<std::uint16_t>::max() ||
        !in.varuint(deadline_delta) || !in.varuint(reason) ||
        reason > std::numeric_limits<std::uint16_t>::max() ||
        !::rund::ValidReasonCode(static_cast<::rund::ReasonCode>(reason))) {
      return false;
    }
    if (!rund::kernel::checked::add(previous_sequence, sequence_delta,
                                    observation.sequence)) {
      return false;
    }
    observation.kind = static_cast<task::ObservationKind>(kind);
    observation.fd = static_cast<int>(fd);
    observation.interest =
        static_cast<short>(static_cast<std::uint16_t>(interest));
    observation.revents =
        static_cast<short>(static_cast<std::uint16_t>(revents));
    std::uint64_t deadline = 0u;
    if (!rund::kernel::checked::add(previous_deadline, deadline_delta,
                                    deadline)) {
      return false;
    }
    observation.deadline_ns = std::bit_cast<std::int64_t>(deadline);
    observation.reason_code = static_cast<::rund::ReasonCode>(reason);
    previous_sequence = observation.sequence;
    previous_deadline = deadline;
  }
  return true;
}

bool write_trace(Writer &out, const ::rund::Trace &trace) noexcept {
  if (!out.ok()) {
    return false;
  }
  if (!::rund::ValidReasonCode(trace.code)) {
    return out.reject(::rund::replay::Code::CodecInvariantInvalid);
  }
  if (!out.varuint(static_cast<std::uint64_t>(trace.code)) ||
      !out.varuint(trace.dropped) ||
      !out.varuint(static_cast<std::uint64_t>(trace.records.size()))) {
    return false;
  }
  std::uint64_t previous_sequence = 0u;
  for (const ::rund::TraceRecord &record : trace.records) {
    std::uint64_t sequence_delta = 0u;
    if (record.event > ::rund::TraceEvent::TelemetrySkipped ||
        !record.code.valid() ||
        record.snapshot.state > ::rund::SessionState::Stopped ||
        !::rund::ValidReasonCode(record.snapshot.code) ||
        !rund::kernel::checked::sub(record.sequence, previous_sequence,
                                    sequence_delta)) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    if (!out.varuint(static_cast<std::uint64_t>(record.event)) ||
        !out.varuint(static_cast<std::uint64_t>(record.code.domain())) ||
        !out.varuint(record.code.value()) ||
        !out.varuint(static_cast<std::uint64_t>(record.snapshot.state)) ||
        !out.varuint(record.snapshot.active_compute_jobs) ||
        !out.u8(static_cast<std::uint8_t>(record.snapshot.scope_active)) ||
        !out.varuint(static_cast<std::uint64_t>(record.snapshot.code)) ||
        !out.varuint(sequence_delta)) {
      return false;
    }
    previous_sequence = record.sequence;
  }
  return true;
}

bool read_trace(Reader &in, Admission &admission, ::rund::Trace &trace) {
  std::uint64_t code = 0u;
  std::uint64_t count = 0u;
  std::size_t count_size = 0u;
  if (!in.varuint(code) || code > std::numeric_limits<std::uint16_t>::max() ||
      !::rund::ValidReasonCode(static_cast<::rund::ReasonCode>(code)) ||
      !in.varuint(trace.dropped) || !in.varuint(count) ||
      !admission.entries(count) || !size(count, count_size)) {
    return false;
  }
  trace.code = static_cast<::rund::ReasonCode>(code);
  trace.records.resize(count_size);
  std::uint64_t previous_sequence = 0u;
  for (::rund::TraceRecord &record : trace.records) {
    std::uint64_t event = 0u;
    std::uint64_t domain = 0u;
    std::uint64_t value = 0u;
    std::uint64_t state = 0u;
    std::uint64_t active = 0u;
    std::uint8_t scope = 0u;
    std::uint64_t snapshot_code = 0u;
    std::uint64_t sequence_delta = 0u;
    if (!in.varuint(event) ||
        event >
            static_cast<std::uint64_t>(::rund::TraceEvent::TelemetrySkipped) ||
        !in.varuint(domain) ||
        domain > static_cast<std::uint64_t>(::rund::TraceDomain::Compute) ||
        !in.varuint(value) ||
        value > std::numeric_limits<std::uint16_t>::max() ||
        !in.varuint(state) ||
        state > static_cast<std::uint64_t>(::rund::SessionState::Stopped) ||
        !in.varuint(active) ||
        active > std::numeric_limits<std::uint32_t>::max() || !in.u8(scope) ||
        scope > 1u || !in.varuint(snapshot_code) ||
        snapshot_code > std::numeric_limits<std::uint16_t>::max() ||
        !::rund::ValidReasonCode(
            static_cast<::rund::ReasonCode>(snapshot_code)) ||
        !in.varuint(sequence_delta)) {
      return false;
    }
    record.event = static_cast<::rund::TraceEvent>(event);
    record.code =
        domain == static_cast<std::uint64_t>(::rund::TraceDomain::Runtime)
            ? ::rund::TraceCode::runtime(static_cast<::rund::ReasonCode>(value))
            : ::rund::TraceCode::compute(
                  static_cast<::rund::compute::Reason>(value));
    if (!record.code.valid()) {
      return false;
    }
    record.snapshot.state = static_cast<::rund::SessionState>(state);
    record.snapshot.active_compute_jobs = static_cast<std::uint32_t>(active);
    record.snapshot.scope_active = scope != 0u;
    record.snapshot.code = static_cast<::rund::ReasonCode>(snapshot_code);
    if (!rund::kernel::checked::add(previous_sequence, sequence_delta,
                                    record.sequence)) {
      return false;
    }
    previous_sequence = record.sequence;
  }
  return true;
}

} // namespace rund::node::replay_detail::artifact
