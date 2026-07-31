#pragma once

#include <accel/device.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

struct FakeAdapter {
  rund::kernel::ComputeCaps caps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 64u * 1024u * 1024u,
      .staging_bytes = 1024u * 1024u,
      .max_window_tiles = 64u,
      .subgroup_width = 1u,
      .ok = true,
      .reason = "ok",
  };
};

[[nodiscard]] bool
ExecuteFake(void *context, const rund::kernel::ComputePlan &plan,
            const rund::kernel::LoweringArtifact &artifact,
            const rund::kernel::ComputeDispatchWindow *windows,
            rund::kernel::u64 window_count,
            const rund::kernel::BindingSet &bindings);

[[nodiscard]] bool ExecuteRetainedFake(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings);

} // namespace rund::node::accel::detail
