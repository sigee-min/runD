#pragma once

#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <string_view>

namespace rund::compute {

class Submission;

class Completion final {
public:
  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status_);
  }
  [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] Reason reason() const noexcept { return status_.reason(); }
  [[nodiscard]] Code code() const noexcept { return status_.code(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }
  [[nodiscard]] int exit_code() const noexcept { return status_.exit_code(); }
  [[nodiscard]] Stats stats() const noexcept { return stats_; }

private:
  friend class Submission;

  Completion(Status status, Stats stats) noexcept
      : status_(status), stats_(stats) {}

  Status status_{Status::fail(Reason::TaskInvalid)};
  Stats stats_{};
};

} // namespace rund::compute
