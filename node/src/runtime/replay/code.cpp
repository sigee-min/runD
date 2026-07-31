#include <node/runtime/replay/code.hpp>

namespace rund::replay {

bool valid(const Code code) noexcept {
  switch (code) {
  case Code::Ok:
    return true;
#define RUND_REPLAY_CODE(domain, value, name, text)                            \
  case Code::name:                                                             \
    return true;
#include <rund/replay/code.def>
#undef RUND_REPLAY_CODE
#define RUND_REPLAY_RUNTIME_Universal(value, name, text)
#define RUND_REPLAY_RUNTIME_General(value, name, text)                         \
  case Code::name:                                                             \
    return true;
#define RUND_REPLAY_RUNTIME_PreparedMemory(value, name, text)                  \
  case Code::name:                                                             \
    return true;
#define RUND_NODE_REASON(value, name, text, category)                          \
  RUND_REPLAY_RUNTIME_##category(value, name, text)
#include <rund/reason.def>
#undef RUND_NODE_REASON
#undef RUND_REPLAY_RUNTIME_PreparedMemory
#undef RUND_REPLAY_RUNTIME_General
#undef RUND_REPLAY_RUNTIME_Universal
  }
  return false;
}

std::string_view error(const Code code) noexcept {
  switch (code) {
  case Code::Ok:
    return {};
#define RUND_REPLAY_CODE(domain, value, name, text)                            \
  case Code::name:                                                             \
    return text;
#include <rund/replay/code.def>
#undef RUND_REPLAY_CODE
#define RUND_REPLAY_RUNTIME_Universal(value, name, text)
#define RUND_REPLAY_RUNTIME_General(value, name, text)                         \
  case Code::name:                                                             \
    return text;
#define RUND_REPLAY_RUNTIME_PreparedMemory(value, name, text)                  \
  case Code::name:                                                             \
    return text;
#define RUND_NODE_REASON(value, name, text, category)                          \
  RUND_REPLAY_RUNTIME_##category(value, name, text)
#include <rund/reason.def>
#undef RUND_NODE_REASON
#undef RUND_REPLAY_RUNTIME_PreparedMemory
#undef RUND_REPLAY_RUNTIME_General
#undef RUND_REPLAY_RUNTIME_Universal
  }
  return "replay_code_unknown";
}

} // namespace rund::replay

namespace rund::node::replay_detail {

::rund::replay::Code code(const ReasonCode reason) noexcept {
  if (reason == ReasonCode::Ok) {
    return ::rund::replay::Code::Ok;
  }
  if (!ValidReasonCode(reason)) {
    return ::rund::replay::Code::CodecInvariantInvalid;
  }
  return static_cast<::rund::replay::Code>((2u << 16u) |
                                           static_cast<std::uint16_t>(reason));
}

std::optional<ReasonCode> reason(const ::rund::replay::Code code) noexcept {
  if (code == ::rund::replay::Code::Ok) {
    return ReasonCode::Ok;
  }
  if (::rund::replay::family(code) != ::rund::replay::Family::Runtime) {
    return std::nullopt;
  }
  const auto reason =
      static_cast<ReasonCode>(::rund::replay::raw(code) & 0xffffu);
  return ValidReasonCode(reason) && reason != ReasonCode::Ok
             ? std::optional<ReasonCode>{reason}
             : std::nullopt;
}

} // namespace rund::node::replay_detail
