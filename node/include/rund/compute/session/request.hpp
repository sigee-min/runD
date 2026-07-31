#pragma once

#include <memory>
#include <utility>

namespace rund::compute {

class Submission;

namespace detail {
struct SessionAccess;
}

class Request final {
public:
  class Awaiter;

  Request() noexcept = default;
  Request(const Request &) = delete;
  Request &operator=(const Request &) = delete;
  Request(Request &&) noexcept = default;
  Request &operator=(Request &&) noexcept = default;

  [[nodiscard]] Submission submit() const noexcept;
  [[nodiscard]] Awaiter operator co_await() && noexcept;

private:
  friend class Awaiter;
  friend struct detail::SessionAccess;

  Request(std::weak_ptr<void> host, std::shared_ptr<void> operation,
          const void *operations) noexcept
      : host_(std::move(host)), operation_(std::move(operation)),
        operations_(operations) {}

  std::weak_ptr<void> host_{};
  std::shared_ptr<void> operation_{};
  const void *operations_ = nullptr;
};

} // namespace rund::compute
