#include "data.hpp"

#include <node/runtime/replay/diff.hpp>
#include <node/runtime/replay/record.hpp>

#include <memory>
#include <utility>

namespace rund::replay {
namespace {

[[nodiscard]] Trace
trace(const node::RuntimeReplayTraceRecordEvidence &value) noexcept {
  return Trace{.event = value.event,
               .code = value.code,
               .snapshot_state = value.snapshot_state,
               .snapshot_active_compute_jobs =
                   value.snapshot_active_compute_jobs,
               .snapshot_scope_active = value.snapshot_scope_active,
               .snapshot_code = value.snapshot_code,
               .sequence = value.sequence};
}

[[nodiscard]] InputPoint point(const node::ReplayInputPoint &value) noexcept {
  return InputPoint{.index = value.index,
                    .input = {.id = value.source, .schema = value.schema},
                    .sequence = value.sequence,
                    .size = value.size,
                    .hash = value.hash};
}

} // namespace

Window::Data::Data(node::RuntimeReplayMismatchWindow value)
    : window(std::move(value)) {
  expected_inputs.reserve(window.expected_inputs.size());
  for (const node::ReplayInputPoint &value : window.expected_inputs) {
    expected_inputs.push_back(point(value));
  }
  actual_inputs.reserve(window.actual_inputs.size());
  for (const node::ReplayInputPoint &value : window.actual_inputs) {
    actual_inputs.push_back(point(value));
  }
  decltype(window.expected_inputs){}.swap(window.expected_inputs);
  decltype(window.actual_inputs){}.swap(window.actual_inputs);

  expected_trace.reserve(window.expected_trace_records.size());
  for (const node::RuntimeReplayTraceRecordEvidence &value :
       window.expected_trace_records) {
    expected_trace.push_back(trace(value));
  }
  actual_trace.reserve(window.actual_trace_records.size());
  for (const node::RuntimeReplayTraceRecordEvidence &value :
       window.actual_trace_records) {
    actual_trace.push_back(trace(value));
  }
  decltype(window.expected_trace_records){}.swap(window.expected_trace_records);
  decltype(window.actual_trace_records){}.swap(window.actual_trace_records);
}

namespace detail {

Check check_result(const Record &expected, Session &session,
                   Session::Result &&actual,
                   const std::uint64_t start_hash) noexcept {
  if (const Code code = ready_code(expected); code != Code::Ok) {
    return fail_check(expected, code);
  }
  Record actual_record = build_record(session, std::move(actual), start_hash);
  if (!actual_record) {
    return Check{actual_record.code(), expected.hash(),
                 std::move(actual_record)};
  }
  try {
    const node::RuntimeReplayCheck checked = node::check_runtime_replay(
        expected.data_->record, actual_record.data_->record);
    return Check{checked.code, expected.hash(), std::move(actual_record)};
  } catch (...) {
    return Check{Code::AllocationFailed, expected.hash(),
                 std::move(actual_record)};
  }
}

Check fail_check(const Record &expected, const Code code) noexcept {
  return Check{code == Code::Ok ? Code::CheckpointInvalid : code,
               expected.hash(), std::nullopt};
}

} // namespace detail

bool Diff::ok() const noexcept { return code() == Code::Ok; }

Code Diff::code() const noexcept { return data_ ? data_->diff.code : code_; }

std::string_view Diff::error() const noexcept {
  return ::rund::replay::error(code());
}

std::size_t Diff::mismatch_count() const noexcept {
  return data_ ? data_->diff.mismatches.size() : 0u;
}

std::optional<Mismatch> Diff::mismatch(const std::size_t index) const noexcept {
  if (!data_ || index >= data_->diff.mismatches.size()) {
    return std::nullopt;
  }
  const node::RuntimeReplayFieldMismatch &value = data_->diff.mismatches[index];
  return Mismatch{.code = value.code,
                  .field = value.field,
                  .expected = value.expected,
                  .actual = value.actual};
}

bool Window::ok() const noexcept { return code() == Code::Ok; }

Code Window::code() const noexcept {
  return data_ ? data_->window.code : code_;
}

std::string_view Window::error() const noexcept {
  return ::rund::replay::error(code());
}

std::optional<std::size_t> Window::observation_index() const noexcept {
  return data_ && data_->window.has_observation_mismatch
             ? std::optional<std::size_t>{data_->window.observation_index}
             : std::nullopt;
}

std::span<const task::Observation>
Window::expected_observations() const noexcept {
  return data_ ? std::span<const task::Observation>{data_->window
                                                        .expected_observations}
               : std::span<const task::Observation>{};
}

std::span<const task::Observation>
Window::actual_observations() const noexcept {
  return data_ ? std::span<const task::Observation>{data_->window
                                                        .actual_observations}
               : std::span<const task::Observation>{};
}

std::optional<std::size_t> Window::host_event_index() const noexcept {
  return data_ && data_->window.has_host_mismatch
             ? std::optional<std::size_t>{data_->window.host_event_index}
             : std::nullopt;
}

std::span<const host::Event> Window::expected_host_events() const noexcept {
  return data_
             ? std::span<const host::Event>{data_->window.expected_host_events}
             : std::span<const host::Event>{};
}

std::span<const host::Event> Window::actual_host_events() const noexcept {
  return data_ ? std::span<const host::Event>{data_->window.actual_host_events}
               : std::span<const host::Event>{};
}

std::optional<std::size_t> Window::input_index() const noexcept {
  return data_ && data_->window.has_input_mismatch
             ? std::optional<std::size_t>{data_->window.input_index}
             : std::nullopt;
}

std::span<const InputPoint> Window::expected_inputs() const noexcept {
  return data_ ? std::span<const InputPoint>{data_->expected_inputs}
               : std::span<const InputPoint>{};
}

std::span<const InputPoint> Window::actual_inputs() const noexcept {
  return data_ ? std::span<const InputPoint>{data_->actual_inputs}
               : std::span<const InputPoint>{};
}

std::optional<std::size_t> Window::trace_record_index() const noexcept {
  return data_ && data_->window.has_trace_mismatch
             ? std::optional<std::size_t>{data_->window.trace_record_index}
             : std::nullopt;
}

std::span<const Trace> Window::expected_trace() const noexcept {
  return data_ ? std::span<const Trace>{data_->expected_trace}
               : std::span<const Trace>{};
}

std::span<const Trace> Window::actual_trace() const noexcept {
  return data_ ? std::span<const Trace>{data_->actual_trace}
               : std::span<const Trace>{};
}

Check check(const Record &expected, const Record &actual) noexcept {
  if (const Code code = detail::ready_code(expected); code != Code::Ok) {
    return Check{code, expected.hash(), actual};
  }
  if (const Code code = detail::ready_code(actual); code != Code::Ok) {
    return Check{code, expected.hash(), actual};
  }
  const node::RuntimeReplayCheck checked =
      node::check_runtime_replay(expected.data_->record, actual.data_->record);
  return Check{checked.code, expected.hash(), actual};
}

Diff diff(const Record &expected, const Record &actual) noexcept {
  try {
    const Code expected_code = detail::ready_code(expected);
    const Code actual_code = detail::ready_code(actual);
    if (expected_code != Code::Ok || actual_code != Code::Ok) {
      return Diff{expected_code != Code::Ok ? expected_code : actual_code};
    }
    return Diff{
        std::make_shared<const Diff::Data>(node::DiffRuntimeReplayRecords(
            expected.data_->record, actual.data_->record))};
  } catch (...) {
    return Diff{Code::AllocationFailed};
  }
}

Window window(const Record &expected, const Record &actual,
              const std::size_t context) noexcept {
  try {
    const Code expected_code = detail::ready_code(expected);
    const Code actual_code = detail::ready_code(actual);
    if (expected_code != Code::Ok || actual_code != Code::Ok) {
      return Window{expected_code != Code::Ok ? expected_code : actual_code};
    }
    return Window{std::make_shared<const Window::Data>(
        node::MinimizeRuntimeReplayMismatch(expected.data_->record,
                                            actual.data_->record, context))};
  } catch (...) {
    return Window{Code::AllocationFailed};
  }
}

} // namespace rund::replay
