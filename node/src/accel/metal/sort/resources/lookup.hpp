#pragma once

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"
#include <accel/check.hpp>
#include <accel/device.hpp>
namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {
[[nodiscard]] bool
MetalSortUsesIdentityValues(const rund::kernel::SortPlan &plan) {
  return plan.value == rund::kernel::SortValue::IdentityU32;
}

[[nodiscard]] const char *
MetalSortLookupReason(const MetalSortEncodeResources &resources,
                      const bool identity_values) {
  if (!resources.read_keys.check.ok) {
    return resources.read_keys.check.reason;
  }
  if (!identity_values && !resources.read_values.check.ok) {
    return resources.read_values.check.reason;
  }
  if (!resources.write_keys.check.ok) {
    return resources.write_keys.check.reason;
  }
  if (!resources.write_values.check.ok) {
    return resources.write_values.check.reason;
  }
  if (!resources.logical_count.check.ok) {
    return resources.logical_count.check.reason;
  }
  return "accel_buffer_unavailable";
}

[[nodiscard]] bool MetalSortLookupOk(const MetalSortEncodeResources &resources,
                                     const bool identity_values) {
  return resources.read_keys.check.ok &&
         (identity_values || resources.read_values.check.ok) &&
         resources.write_keys.check.ok && resources.write_values.check.ok &&
         resources.logical_count.check.ok &&
         resources.read_keys.device_buffer != nullptr &&
         (identity_values || resources.read_values.device_buffer != nullptr) &&
         resources.write_keys.device_buffer != nullptr &&
         resources.write_values.device_buffer != nullptr &&
         resources.logical_count.device_buffer != nullptr;
}

[[nodiscard]] rund::AccelCheck
LookupMetalSortBuffers(const rund::AccelDevice &pick,
                       const SortBinds &bindings,
                       MetalSortEncodeResources &resources) {
  const bool identity_values = MetalSortUsesIdentityValues(resources.plan);
  MetalResidentReq reqs[5] = {
      {bindings.read_keys, bindings.read_keys_handle, &resources.read_keys},
      {bindings.write_keys, bindings.write_keys_handle,
       &resources.write_keys},
      {bindings.write_values, bindings.write_values_handle,
       &resources.write_values},
      {},
      {}};
  std::size_t count = 3u;
  if (!identity_values) {
    reqs[1] = {bindings.read_values, bindings.read_values_handle,
               &resources.read_values};
    reqs[2] = {bindings.write_keys, bindings.write_keys_handle,
               &resources.write_keys};
    reqs[3] = {bindings.write_values, bindings.write_values_handle,
               &resources.write_values};
    count = 4u;
  }
  const bool bounded = resources.plan.count_source !=
                       rund::kernel::ComputeCountSource::Descriptor;
  if (bounded) {
    reqs[count++] = {bindings.logical_count, bindings.logical_count_handle,
                     &resources.logical_count};
  }
  LookupMetalResidentBatch(pick, reqs, count,
                           "accel_metal_resident_id_unavailable");
  if (!bounded) {
    resources.logical_count = resources.read_keys;
  }
  if (MetalSortLookupOk(resources, identity_values)) {
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false,
                          MetalSortLookupReason(resources, identity_values)};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
