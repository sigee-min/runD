#pragma once

#include <rund/compute/session/submission.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail {

struct SessionAccess final {
  [[nodiscard]] static Request make(std::weak_ptr<void> host,
                                    std::shared_ptr<void> operation,
                                    const void *operations) noexcept {
    return Request{std::move(host), std::move(operation), operations};
  }
};

} // namespace rund::compute::detail
