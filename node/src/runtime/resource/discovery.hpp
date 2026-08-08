#pragma once

#include <node/resource/envelope.hpp>
#include <node/runtime/backend.hpp>

namespace rund::node::runtime_detail::resource {

struct Request final {
  bool require_verified_numa = false;
  bool require_verified_affinity = false;
  bool require_verified_capacity = false;
};

[[nodiscard]] ResourceEnvelope Resolve(BackendSelection selection,
                                       const Request &request);

} // namespace rund::node::runtime_detail::resource
