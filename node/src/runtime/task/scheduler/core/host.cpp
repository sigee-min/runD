#include "../host.hpp"
#include "../state/model/context.hpp"
#include "../state/storage.hpp"

namespace rund::node::scheduler_host {

bool ActiveTask() noexcept {
  return active_scheduler_context != nullptr &&
         active_scheduler_context->task_id != 0u;
}

std::int64_t LogicalTimeNs() noexcept {
  const Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr ? 0 : scheduler->LogicalTimeNs();
}

::rund::host::random::RunSeed RandomSeed() noexcept {
  const Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr ? ::rund::host::random::RunSeed{}
                              : scheduler->RandomSeed();
}

bool CapturesIngress() noexcept {
  const Scheduler *const scheduler = Scheduler::Active();
  return scheduler != nullptr && scheduler->CapturesNetIngress();
}

bool Record(const ::rund::host::Event event) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler != nullptr && scheduler->RecordHostEventFromHostApi(event);
}

bool Record(const ::rund::host::Event event,
            const replay_detail::payload::RawByteSource &source) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler != nullptr &&
         scheduler->RecordHostEventFromHostApi(event, source);
}

ReplayInputMode InputMode() noexcept {
  const Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr ? ReplayInputMode::Unavailable
                              : scheduler->ReplayInputMode();
}

ReplayInputCapture
BeginInput(const replay_detail::payload::InputBinding &binding) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr
             ? ReplayInputCapture{.code =
                                      ::rund::replay::Code::InputContextMissing}
             : scheduler->BeginReplayInput(binding);
}

void FailInput(const ::rund::replay::Code code) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  if (scheduler != nullptr) {
    scheduler->FailReplayInput(code);
  }
}

void CancelInput(const ReplayInputCapture capture) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  if (scheduler != nullptr) {
    scheduler->CancelReplayInput(capture);
  }
}

replay_detail::payload::ResolveResult
FinishInput(const replay_detail::payload::InputBinding &binding,
            const ReplayInputCapture capture,
            const std::size_t byte_count) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr
             ? replay_detail::payload::
                   ResolveResult{.code =
                                     ::rund::replay::Code::InputContextMissing}
             : scheduler->FinishReplayInput(binding, capture, byte_count);
}

replay_detail::payload::ResolveResult
RejectInput(const ReplayInputCapture capture,
            const ::rund::replay::Code code) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr
             ? replay_detail::payload::
                   ResolveResult{.code =
                                     ::rund::replay::Code::InputContextMissing}
             : scheduler->RejectReplayInput(capture, code);
}

replay_detail::payload::ResolveResult
ReplayInput(const replay_detail::payload::InputBinding &binding) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  return scheduler == nullptr
             ? replay_detail::payload::
                   ResolveResult{.code =
                                     ::rund::replay::Code::InputContextMissing}
             : scheduler->ReplayInput(binding);
}

} // namespace rund::node::scheduler_host
