#include <rund/host/env.hpp>
#include <rund/host/event.hpp>

#include "../runtime/task/scheduler/host.hpp"

#include <cstdlib>
#include <string>
#include <utility>

namespace rund::host::env {
namespace {

[[nodiscard]] task::Result<std::string> FailClosed() noexcept {
  return task::Result<std::string>::fail(ReasonCode::TaskInvalid);
}

} // namespace

task::Result<std::string> get(const std::string_view name) noexcept {
  if (name.empty() || name.find('\0') != std::string_view::npos) {
    return FailClosed();
  }

  std::string name_copy{};
  try {
    name_copy.assign(name.data(), name.size());
  } catch (...) {
    return FailClosed();
  }

  const char *const raw_value = std::getenv(name_copy.c_str());
  std::string value{};
  if (raw_value != nullptr) {
    try {
      value.assign(raw_value);
    } catch (...) {
      return FailClosed();
    }
  }

  (void)::rund::node::scheduler_host::Record(::rund::host::Event{
      .kind = ::rund::host::EventKind::EnvGet,
      .status = ::rund::host::Status::Ok,
      .name_hash = ::rund::host::hash_string(name.data(), name.size()),
      .payload_hash = ::rund::host::hash_string(value.data(), value.size()),
  });

  return task::Result<std::string>::success(std::move(value));
}

} // namespace rund::host::env
