#pragma once

#include <rund/task/handle.hpp>

#include <rund/compute/session/completion.hpp>
#include <rund/compute/session/poll.hpp>
#include <rund/compute/session/request.hpp>

#include <chrono>
#include <memory>
#include <utility>

namespace rund::compute {

class Submission final {
public:
  Submission() noexcept = default;
  ~Submission();
  Submission(const Submission &) = delete;
  Submission &operator=(const Submission &) = delete;
  Submission(Submission &&other) noexcept;
  Submission &operator=(Submission &&other) noexcept;

  [[nodiscard]] Poll poll() const noexcept;
  [[nodiscard]] Poll wait_for(std::chrono::nanoseconds timeout) const noexcept;
  [[nodiscard]] Completion wait() const noexcept;
  [[nodiscard]] Status cancel() const noexcept;

private:
  friend class Request;
  friend class Request::Awaiter;

  Submission(std::shared_ptr<void> host, void *state) noexcept
      : host_(std::move(host)), state_(state) {}
  explicit Submission(Status immediate) noexcept : immediate_(immediate) {}

  [[nodiscard]] task::Handle handle() const noexcept;

  std::shared_ptr<void> host_{};
  void *state_ = nullptr;
  Status immediate_{Status::fail(Reason::TaskInvalid)};
};

} // namespace rund::compute
