#include <rund/session/run.hpp>

#include "result.hpp"

namespace rund::detail::run {
namespace {

struct Call final {
  Session *session = nullptr;
  void *callback = nullptr;
  Access::Callback invoke = nullptr;
};

void invoke(void *const raw) {
  auto &call = *static_cast<Call *>(raw);
  call.invoke(call.callback, *call.session);
}

} // namespace

Session::Result Access::execute(SessionConfig config, void *const callback,
                                const Callback invoke_callback) {
  Session session{};
  const Session::Status opened = session.open(std::move(config));
  if (!opened) {
    return ::rund::detail::session::ResultAccess::fail(opened.code(),
                                                    session.take_trace());
  }

  Call call{
      .session = &session, .callback = callback, .invoke = invoke_callback};
  Session::Result result = session.terminal(&call, invoke);
  const bool scope_ok = result.ok();
  const ::rund::ReasonCode scope_code = result.code();
  const Session::Status closed = session.close();

  ::rund::ReasonCode code = ::rund::ReasonCode::Ok;
  if (!scope_ok) {
    code = scope_code;
  } else if (!closed) {
    code = closed.code();
  }
  ::rund::detail::session::ResultAccess::finish(result, code,
                                             session.take_trace());
  return result;
}

} // namespace rund::detail::run
