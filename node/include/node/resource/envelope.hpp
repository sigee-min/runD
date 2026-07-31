#pragma once

#include <kernel/dispatch/worker/backend.hpp>
#include <rund/session/resources.hpp>

namespace rund::node {

struct ResourceEnvelope {
  ::rund::Resources observed{};
  rund::kernel::WorkerBackend worker_backend{};
};

} // namespace rund::node
