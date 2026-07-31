#pragma once

#include <accel/graph/factory/sort/values.hpp>
#include <kernel/program/compute/sort/plan.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "test/compute/fixed.hpp"

#include "../identity.hpp"

namespace node_accel_contract::sort_identity {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read_keys{};
  rund::AccelBuffer write_keys{};
  rund::AccelBuffer write_values{};
  rund::kernel::SortPlan sort_plan{};
  rund::AccelKernel kernel{};
  bool valid{false};
};

[[nodiscard]] inline Resources
PrepareResources(const rund::AccelDevice &pick,
                 const rund::kernel::ComputeScalar scalar,
                 const std::array<rund::kernel::u32, 8u> &input_keys,
                 const rund::kernel::u32 key_bits) {
  namespace p = node_accel_contract::primitive;
  Resources resources{};
  resources.context = rund::node::accel::OpenAccel(pick);
  if (!resources.context.check.ok) {
    return resources;
  }

  resources.read_keys = rund::node::accel::CreateAccelBuffer(
      resources.context,
      p::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                    input_keys.size()));
  resources.write_keys = rund::node::accel::CreateAccelBuffer(
      resources.context,
      p::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32),
                    input_keys.size()));
  resources.write_values = rund::node::accel::CreateAccelBuffer(
      resources.context,
      p::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32),
                    input_keys.size()));
  if (!resources.read_keys.check.ok || !resources.write_keys.check.ok ||
      !resources.write_values.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           resources.context, resources.read_keys, input_keys.data(),
           input_keys.size() * sizeof(rund::kernel::u32))
           .ok) {
    return resources;
  }

  std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &resources.read_keys,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &resources.write_keys,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelGraphBufferRef{
          .buffer = &resources.write_values,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::SortDesc desc{
      .key = rund::kernel::SortKey::U32,
      .value = rund::kernel::SortValue::IdentityU32,
      .element_count = input_keys.size(),
      .radix_bits = 8u,
      .key_bits = key_bits,
      .stable = true,
  };
  resources.sort_plan = rund::kernel::PlanSort(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSort(refs.data(), refs.size(), desc)};
  resources.kernel = rund::node::accel::CompileAccelKernel(
      resources.context, rund::AccelGraph{
                             .nodes = nodes.data(),
                             .node_count = nodes.size(),
                             .scalar = scalar,
                             .domain = rund::kernel::ComputeDomain::Fixed,
                             .fixed_format = test::FixedFormatForLane(scalar),
                         });
  resources.valid = resources.sort_plan.ok && resources.kernel.check.ok;
  return resources;
}

} // namespace node_accel_contract::sort_identity
