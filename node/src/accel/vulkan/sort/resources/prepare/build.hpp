#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

inline rund::AccelCheck BuildVulkanSortResources(
    VulkanAdapter *const adapter, const rund::AccelDevice &pick,
    const rund::kernel::SortDesc &desc, const rund::kernel::SortPlan &plan,
    const rund::kernel::ComputeDomain domain, const SortBinds &bindings,
    std::shared_ptr<void> &resources,
    const VulkanKernelImmutablePipelines *const pipelines) {
  VulkanSortPrepareShape shape{};
  const rund::AccelCheck shape_check =
      ValidateVulkanSortPrepareShape(*adapter, desc, plan, bindings, shape);
  if (!shape_check.ok) {
    return shape_check;
  }

  VulkanSortResidentBuffers buffers{};
  const rund::AccelCheck lookup =
      LookupVulkanSortResidentBuffers(*adapter, pick, plan, bindings, buffers);
  if (!lookup.ok) {
    return lookup;
  }

  auto *const raw = new VulkanSortEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanSortEncodeResources};
  raw->adapter = adapter;
  raw->logical_count = buffers.logical_count;
  const rund::AccelCheck allocated =
      AllocateVulkanSortSharedResources(*adapter, desc, plan, shape, *raw,
                                        pipelines);
  if (!allocated.ok) {
    return allocated;
  }

  SortParams params_values[kMaxSortPasses]{};
  const rund::AccelCheck dispatch =
      PrepareVulkanSortDispatch(*adapter, buffers, *raw);
  if (!dispatch.ok) {
    return dispatch;
  }
  const bool signed_order = IsSignedDomain(domain);
  for (std::size_t pass = 0u; pass < raw->pass_count; ++pass) {
    const rund::AccelCheck prepared = PrepareVulkanSortPass(
        *adapter, plan, buffers, signed_order, pass, *raw, params_values[pass]);
    if (!prepared.ok) {
      return prepared;
    }
  }
  std::vector<std::byte> params_bytes;
  try {
    params_bytes.resize(raw->pass_count * raw->params_stride);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t pass = 0u; pass < raw->pass_count; ++pass) {
    std::memcpy(params_bytes.data() + pass * raw->params_stride,
                &params_values[pass], sizeof(SortParams));
  }
  if (!UploadVulkanBuffer(raw->params, params_bytes.data(),
                          params_bytes.size())) {
    SetVulkanLastError(*adapter, "accel_vulkan_memory_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
}
