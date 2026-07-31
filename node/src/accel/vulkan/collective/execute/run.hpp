#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

template <class Prepare, class... Args>
rund::AccelCheck
ExecuteVulkanPrepared(const rund::AccelDevice &pick, Prepare prepare,
                      const VulkanEncodedResourceFn encode,
                      const VulkanFinishResourceFn finish, Args &&...args) {
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  BeginVulkanCollectiveDescriptorEpoch(*adapter);
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepared =
      prepare(pick, std::forward<Args>(args)..., resources);
  if (!prepared.ok) {
    return prepared;
  }
  return SubmitVulkanEncodedResources(*adapter, resources, encode, finish);
}

template <class Desc, class Plan, class Bindings, class Prepare>
rund::AccelCheck
ExecuteVulkanCollective(const rund::AccelDevice &pick, const Desc &desc,
                        const Plan &plan, const Bindings &bindings,
                        Prepare prepare, const VulkanEncodedResourceFn encode,
                        const VulkanFinishResourceFn finish) {
  return ExecuteVulkanPrepared(pick, prepare, encode, finish, desc, plan,
                               bindings);
}

template <class Desc, class Plan, class Bindings, class Prepare>
rund::AccelCheck ExecuteVulkanDomainCollective(
    const rund::AccelDevice &pick, const Desc &desc, const Plan &plan,
    const rund::kernel::ComputeDomain domain, const Bindings &bindings,
    Prepare prepare, const VulkanEncodedResourceFn encode,
    const VulkanFinishResourceFn finish) {
  return ExecuteVulkanPrepared(pick, prepare, encode, finish, desc, plan,
                               domain, bindings);
}
