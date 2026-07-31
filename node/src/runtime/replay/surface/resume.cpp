#include "local.hpp"

#include "../scope/timing.hpp"

#include <memory>
#include <utility>

namespace rund::replay::detail::surface {

Code restore(const Checkpoint &checkpoint, const RestoreCall call) noexcept {
  if (call.invoke == nullptr || call.object == nullptr) {
    return Code::CheckpointRestoreInvalid;
  }
  try {
    return call.invoke(call.object, checkpoint.state()) == Restore::Restored
               ? Code::Ok
               : Code::CheckpointRestoreFailed;
  } catch (...) {
    return Code::CheckpointRestoreThrew;
  }
}

} // namespace rund::replay::detail::surface

namespace rund::replay::detail {

Record Access::resume_record(Session &session, const Resume &resume,
                             const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  if (resume.code_ != Code::Ok) {
    return reject(session, resume.code_, timing);
  }
  if (const Code code = surface::restore(resume.checkpoint_, resume.restore_);
      code != Code::Ok) {
    return reject(session, code, timing);
  }
  return capture(session, checkpoint_start(resume.checkpoint_.hash()), call,
                 timing);
}

Check Access::resume_run(Session &session, const Resume &resume,
                         const Record &expected_record, const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  telemetry::Preparation plan = telemetry::Preparation::None;
  if (resume.code_ != Code::Ok) {
    return reject(session, expected_record, resume.code_, plan, timing);
  }
  const std::uint64_t start = checkpoint_start(resume.checkpoint_.hash());
  Code code = Code::Ok;
  std::shared_ptr<const void> prepared =
      expected(session, expected_record, start, code, plan);
  if (!prepared) {
    return reject(session, expected_record, code, plan, timing);
  }
  if (const Code restored =
          surface::restore(resume.checkpoint_, resume.restore_);
      restored != Code::Ok) {
    return reject(session, expected_record, restored, plan, timing);
  }
  return replay(session, expected_record, start, call, std::move(prepared),
                plan, timing);
}

Scenario Access::resume_scenario(Session &session, const Resume &resume,
                                 const Record &expected_record,
                                 const std::span<const Choice> choices,
                                 const Call call) {
  scope::Timing timing{scope::Access::detail(session)};
  telemetry::Preparation preparation = telemetry::Preparation::None;
  const std::uint64_t choice_count = surface::count(choices.size());
  if (resume.code_ != Code::Ok) {
    return reject(session, resume.code_, preparation, choice_count, timing);
  }
  const std::uint64_t start = checkpoint_start(resume.checkpoint_.hash());
  Code code = Code::Ok;
  std::shared_ptr<const void> prepared =
      expected(session, expected_record, start, code, preparation);
  if (!prepared) {
    return reject(session, code, preparation, choice_count, timing);
  }
  ReplayScenarioPlan plan = prepare_scenario(session, expected_record, choices);
  if (!plan.ok()) {
    return reject(session, plan.code, preparation, choice_count, timing);
  }
  if (const Code restored =
          surface::restore(resume.checkpoint_, resume.restore_);
      restored != Code::Ok) {
    return reject(session, restored, preparation, choice_count, timing);
  }
  return explore(session, expected_record, choices, start, call,
                 std::move(prepared), preparation, std::move(plan), timing);
}

} // namespace rund::replay::detail
