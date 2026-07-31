#include <rund/replay/input.hpp>
#include <rund/telemetry/event.hpp>

namespace rund::replay {

Value Channel::read(Context &context) const {
  return ok() ? context.read(input_, source_) : context.reject(code_);
}

Choice Channel::choice(const std::uint64_t sequence,
                       const std::span<const std::byte> bytes) const noexcept {
  return Choice{input_, sequence, bytes};
}

Value Context::read(const Input input, const detail::SourceCall source) {
  if (!lease_.valid()) {
    return Value{{}, 0u, Code::ScopeExpired, lease_};
  }
  if (mode_ == telemetry::Mode::Replay || mode_ == telemetry::Mode::Scenario) {
    return detail::replay_input(input, lease_);
  }
  const detail::ReplayInputCapture capture = detail::begin_input(input);
  const detail::Request rejected{.input = input};
  if (!capture.ok()) {
    return detail::finish_input(rejected, capture, 0u, lease_);
  }
  Writer writer{capture.bytes};
  try {
    const std::uint64_t sequence = source.invoke(source.object, writer);
    if (!writer.finish()) {
      return detail::reject_input(capture, writer.code(), lease_);
    }
    return detail::finish_input(
        detail::Request{.input = input, .sequence = sequence}, capture,
        writer.size(), lease_);
  } catch (...) {
    detail::cancel_input(capture);
    throw;
  }
}

Value Context::reject(const Code code) const noexcept {
  detail::fail_input(code);
  return Value{{}, 0u, code, lease_};
}

} // namespace rund::replay
