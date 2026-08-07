#include "local.hpp"

#include "../exception.hpp"
#include "../input/plan.hpp"
#include "../scope/timing.hpp"

#include <node/runtime/replay/code.hpp>

#include <memory>
#include <mutex>
#include <utility>

namespace rund::replay {

Code Live::code() const noexcept {
  return node::replay_detail::code(result_.code());
}

namespace detail {

Session::Result Access::execute(Session &session, const telemetry::Mode mode,
                                std::shared_ptr<const void> expected,
                                std::shared_ptr<const void> choices,
                                const Call call, bool &callback_ran,
                                scope::Timing &timing) {
  const auto prepared =
      std::static_pointer_cast<const scope::Expected>(std::move(expected));
  const auto input =
      std::static_pointer_cast<const node::replay_detail::InputPlan>(
          std::move(choices));
  scope::Mode scope_mode = scope::Mode::Live;
  switch (mode) {
  case telemetry::Mode::Live:
    scope_mode = scope::Mode::Live;
    break;
  case telemetry::Mode::Record:
    scope_mode = scope::Mode::Record;
    break;
  case telemetry::Mode::Replay:
    scope_mode = scope::Mode::Replay;
    break;
  case telemetry::Mode::Scenario:
    scope_mode = scope::Mode::Scenario;
    break;
  case telemetry::Mode::None:
    return Session::Result{};
  }
  struct Bound final {
    Call call{};
    telemetry::Mode mode = telemetry::Mode::None;
    bool *ran = nullptr;
  } bound{.call = call, .mode = mode, .ran = &callback_ran};
  const auto invoke =
      +[](void *const raw, Session &active, const scope::Lease lease) {
        auto &bound = *static_cast<Bound *>(raw);
        *bound.ran = true;
        Context context{
            bound.mode,
            Lease{.generation = lease.generation, .value = lease.value},
        };
        bound.call.invoke(bound.call.object, context, active);
      };
  return scope::Access::run(
      session,
      scope::Plan{.mode = scope_mode, .expected = prepared, .choices = input},
      &bound, invoke, timing);
}

std::shared_ptr<const void>
Access::expected(Session &session, const Record &record,
                 const std::uint64_t start, Code &code,
                 telemetry::Preparation &plan) noexcept {
  plan = telemetry::Preparation::None;
  if (const Code ready = ready_code(record); ready != Code::Ok) {
    code = ready;
    return {};
  }
  if (record.start_hash() != start) {
    code = Code::RecordStartMismatch;
    return {};
  }
  if (record.data_->record.input_count > scope::Access::inputs(session)) {
    code = Code::InputCapacityExceeded;
    return {};
  }
  if (record.data_->published.load(std::memory_order_acquire) != nullptr) {
    plan = telemetry::Preparation::Reused;
    code = Code::Ok;
    return std::static_pointer_cast<const void>(record.data_->prepared);
  }
  try {
    std::lock_guard lock{record.data_->prepare_mutex};
    if (record.data_->published.load(std::memory_order_relaxed) == nullptr) {
      scope::Prepared prepared =
          scope::Access::prepare(session, record.data_->record.host.events,
                                 record.data_->record.host.payload_archive);
      if (!prepared) {
        code = prepared.code;
        return {};
      }
      record.data_->prepared = std::move(prepared.owner);
      record.data_->published.store(record.data_->prepared.get(),
                                    std::memory_order_release);
      plan = telemetry::Preparation::Built;
    } else {
      plan = telemetry::Preparation::Reused;
    }
    code = Code::Ok;
    return std::static_pointer_cast<const void>(record.data_->prepared);
  } catch (...) {
    code = node::replay_detail::CurrentExceptionCode({
        .bad_alloc = Code::AllocationFailed,
        .length_error = Code::ScopePrepareFailed,
        .unexpected = Code::ScopePrepareFailed,
    });
    return {};
  }
}

Live Access::live(Session &session, const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  bool callback_ran = false;
  Session::Result result = execute(session, telemetry::Mode::Live, {}, {}, call,
                                   callback_ran, timing);
  telemetry::Event observed =
      event(telemetry::Mode::Live, telemetry::Preparation::None, 0u, result);
  surface::publish(timing, observed, &result);
  Live output{std::move(result)};
  observed.replay.code = output.code();
  emit(session, std::move(observed));
  return output;
}

Record Access::record(Session &session, const std::uint64_t start,
                      const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  return capture(session, start, call, timing);
}

Record Access::capture(Session &session, const std::uint64_t start,
                       const Call call, scope::Timing &timing) {
  bool callback_ran = false;
  Session::Result result = execute(session, telemetry::Mode::Record, {}, {},
                                   call, callback_ran, timing);
  telemetry::Event observed =
      event(telemetry::Mode::Record, telemetry::Preparation::None, 0u, result);
  Record output = build_record(session, std::move(result), start);
  observed.replay.code = output.code();
  observed.replay.plan = output.data_ && output.data_->prepared
                             ? telemetry::Preparation::Built
                             : telemetry::Preparation::None;
  evidence(observed, output);
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

Check Access::replay(Session &session, const Record &expected_record,
                     const std::uint64_t start, const Call call,
                     std::shared_ptr<const void> prepared,
                     const telemetry::Preparation plan, scope::Timing &timing) {
  bool callback_ran = false;
  Session::Result result =
      execute(session, telemetry::Mode::Replay, std::move(prepared), {}, call,
              callback_ran, timing);
  telemetry::Event observed = event(telemetry::Mode::Replay, plan, 0u, result);
  Check output =
      check_result(expected_record, session, std::move(result), start);
  observed.replay.code = output.code();
  if (output.actual()) {
    evidence(observed, *output.actual());
  }
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

Check Access::run(Session &session, const Record &expected_record,
                  const std::uint64_t start, const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  Code code = Code::Ok;
  telemetry::Preparation plan = telemetry::Preparation::None;
  std::shared_ptr<const void> prepared =
      expected(session, expected_record, start, code, plan);
  if (!prepared) {
    return reject(session, expected_record, code, plan, timing);
  }
  return replay(session, expected_record, start, call, std::move(prepared),
                plan, timing);
}

Scenario Access::explore(Session &session, const Record &expected_record,
                         const std::span<const Choice> choices,
                         const std::uint64_t start, const Call call,
                         std::shared_ptr<const void> prepared,
                         const telemetry::Preparation preparation,
                         ReplayScenarioPlan plan, scope::Timing &timing) {
  bool callback_ran = false;
  Session::Result result =
      execute(session, telemetry::Mode::Scenario, std::move(prepared),
              std::move(plan.choices), call, callback_ran, timing);
  telemetry::Event observed = event(telemetry::Mode::Scenario, preparation,
                                    surface::count(choices.size()), result);
  Scenario output = finish_scenario(expected_record, session, std::move(result),
                                    callback_ran, start);
  observed.replay.code = output.code();
  if (output.actual()) {
    evidence(observed, *output.actual());
  }
  surface::publish(timing, observed);
  emit(session, std::move(observed));
  return output;
}

Scenario Access::scenario(Session &session, const Record &expected_record,
                          const std::span<const Choice> choices,
                          const std::uint64_t start, const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  const std::uint64_t choice_count = surface::count(choices.size());
  Code code = Code::Ok;
  telemetry::Preparation preparation = telemetry::Preparation::None;
  std::shared_ptr<const void> prepared =
      expected(session, expected_record, start, code, preparation);
  if (!prepared) {
    return reject(session, code, preparation, choice_count, timing);
  }
  ReplayScenarioPlan plan = prepare_scenario(session, expected_record, choices);
  if (!plan.ok()) {
    return reject(session, plan.code, preparation, choice_count, timing);
  }
  return explore(session, expected_record, choices, start, call,
                 std::move(prepared), preparation, std::move(plan), timing);
}

} // namespace detail
} // namespace rund::replay
