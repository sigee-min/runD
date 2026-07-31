#include <rund/replay/input.hpp>

#include <node/runtime/replay/hash.hpp>

#include "../../task/scheduler/host.hpp"
#include "../host/payload/store.hpp"

namespace rund::replay::detail {

std::uint64_t genesis_start() noexcept {
  return node::replay_detail::GenesisStartHash();
}

std::uint64_t checkpoint_start(const std::uint64_t checkpoint_hash) noexcept {
  return node::replay_detail::StartHash(checkpoint_hash);
}

ReplayInputCapture begin_input(const Input input) noexcept {
  const node::scheduler_host::ReplayInputMode mode =
      node::scheduler_host::InputMode();
  if (mode != node::scheduler_host::ReplayInputMode::Live &&
      mode != node::scheduler_host::ReplayInputMode::Record) {
    return ReplayInputCapture{.code = Code::InputContextMissing};
  }
  const node::scheduler_host::ReplayInputCapture capture =
      node::scheduler_host::BeginInput(
          node::replay_detail::payload::InputBinding{
              .source = input.id,
              .schema = input.schema,
          });
  return ReplayInputCapture{
      .token = capture.token, .code = capture.code, .bytes = capture.bytes};
}

void cancel_input(const ReplayInputCapture capture) noexcept {
  node::scheduler_host::CancelInput(node::scheduler_host::ReplayInputCapture{
      .token = capture.token,
      .code = capture.code,
  });
}

Value finish_input(const Request request, const ReplayInputCapture capture,
                   const std::size_t byte_count, const Lease lease) {
  if (!capture.ok()) {
    return Value{{}, request.sequence, capture.code, lease};
  }
  node::replay_detail::payload::ResolveResult stored =
      node::scheduler_host::FinishInput(
          node::replay_detail::payload::InputBinding{
              .source = request.input.id,
              .schema = request.input.schema,
              .sequence = request.sequence,
          },
          node::scheduler_host::ReplayInputCapture{
              .token = capture.token,
              .code = capture.code,
          },
          byte_count);
  return stored.ok()
             ? Value{stored.bytes.span(), stored.sequence, Code::Ok, lease}
             : Value{{}, request.sequence, stored.code, lease};
}

Value reject_input(const ReplayInputCapture capture, const Code code,
                   const Lease lease) noexcept {
  node::replay_detail::payload::ResolveResult rejected =
      node::scheduler_host::RejectInput(
          node::scheduler_host::ReplayInputCapture{
              .token = capture.token,
              .code = capture.code,
              .bytes = capture.bytes,
          },
          code);
  return Value{{}, 0u, rejected.code, lease};
}

void fail_input(const Code code) noexcept {
  node::scheduler_host::FailInput(code);
}

Value replay_input(const Input input, const Lease lease) {
  const node::scheduler_host::ReplayInputMode mode =
      node::scheduler_host::InputMode();
  if (mode != node::scheduler_host::ReplayInputMode::Replay &&
      mode != node::scheduler_host::ReplayInputMode::Scenario) {
    return Value{{}, 0u, Code::InputContextMissing, lease};
  }
  node::replay_detail::payload::ResolveResult resolved =
      node::scheduler_host::ReplayInput(
          node::replay_detail::payload::InputBinding{
              .source = input.id,
              .schema = input.schema,
          });
  return resolved.ok()
             ? Value{resolved.bytes.span(), resolved.sequence, Code::Ok, lease}
             : Value{{}, 0u, resolved.code, lease};
}

} // namespace rund::replay::detail
