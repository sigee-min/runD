#pragma once

#include <rund/task/cancel/identity.hpp>

namespace rund::node {

struct StopSourceRecord {
  ::rund::detail::task::StopIdentity identity{};
  bool requested = false;
};

} // namespace rund::node
