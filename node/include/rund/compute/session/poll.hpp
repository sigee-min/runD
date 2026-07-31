#pragma once

#include <rund/compute/status.hpp>

#include <string_view>

namespace rund::compute {

class Submission;

struct Poll final {
  bool submitted = false;
  bool backend_submitted = false;
  bool completed = false;

  [[nodiscard]] constexpr Reason reason() const noexcept {
    return status_.reason();
  }
  [[nodiscard]] constexpr Code code() const noexcept { return status_.code(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return status_.error();
  }

private:
  friend class Submission;

  constexpr Poll(const bool submitted, const bool backend_submitted,
                 const bool completed, const Reason reason) noexcept
      : submitted(submitted), backend_submitted(backend_submitted),
        completed(completed),
        status_(reason == Reason::Ok ? Status::success()
                                     : Status::fail(reason)) {}

  Status status_{Status::fail(Reason::TaskInvalid)};
};

} // namespace rund::compute
