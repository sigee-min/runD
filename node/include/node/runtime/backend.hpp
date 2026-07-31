#pragma once

#include <kernel/dispatch/worker/backend.hpp>
#include <rund/reason.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

namespace rund::node {

struct BackendSelection {
  ReasonCode code = ReasonCode::BackendInvalid;
  std::uint32_t requested_worker_width = 1u;
  rund::kernel::WorkerBackend backend{};
  std::shared_ptr<void> owner{};
  bool verified_numa = false;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : std::string_view{ReasonString(code)};
  }
};

[[nodiscard]] BackendSelection select_backend(std::uint32_t workers);
[[nodiscard]] BackendSelection select_backend(kernel::WorkerBackend backend,
                                              std::uint32_t workers);

} // namespace rund::node
