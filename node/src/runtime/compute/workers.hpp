#pragma once

#include <kernel/dispatch/worker/backend.hpp>

namespace rund::node::runtime_detail {

struct ComputeHostState;

[[nodiscard]] kernel::WorkerBackend
MakeComputeWorkerBackend(ComputeHostState& host) noexcept;

[[nodiscard]] kernel::WorkerBackend
MakeComputeSyncWorkerBackend(ComputeHostState& host) noexcept;

}  // namespace rund::node::runtime_detail
