#pragma once

#include <node/resource/envelope.hpp>
#include <node/runtime/backend.hpp>

namespace rund::node::runtime_detail::resource {

struct Request final {
  std::uint32_t workers = 1u;
  bool require_verified_numa = false;
  bool require_verified_affinity = false;
  bool require_verified_capacity = false;
};

struct Result final {
  ReasonCode code = ReasonCode::RuntimeResourcesInvalid;
  ResourceEnvelope resources{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
};

[[nodiscard]] Result Resolve(BackendSelection selection,
                             const Request &request);

} // namespace rund::node::runtime_detail::resource
