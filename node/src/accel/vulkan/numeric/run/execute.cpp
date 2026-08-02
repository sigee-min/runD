#include "../../numeric.hpp"

#include "../../adapter/api.hpp"
#include "../../collective/finish.hpp"
#include "../../collective/pipeline.hpp"
#include "../../command.hpp"

#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

namespace {
template <class Desc, class Plan, class Bindings, class Prepare>
rund::AccelCheck
ExecutePreparedVulkanNumeric(const rund::AccelDevice &pick, const Desc &desc,
                             const Plan &plan, const Bindings &bindings,
                             Prepare prepare) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  BeginVulkanCollectiveDescriptorEpoch(*adapter);
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check = prepare(
      pick, desc, plan, bindings, KernelPreparationMode::Standalone, prepared,
      nullptr);
  return check.ok ? SubmitVulkanEncodedResources(*adapter, prepared,
                                                 EncodeVulkanNumeric,
                                                 FinishVulkanNumeric)
                  : check;
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)prepare;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}
} // namespace

rund::AccelCheck ExecuteVulkanMatrix(const rund::AccelDevice &pick,
                                     const rund::kernel::MatrixDesc &desc,
                                     const rund::kernel::MatrixPlan &plan,
                                     const MatrixBinds &bindings) {
  return ExecutePreparedVulkanNumeric(pick, desc, plan, bindings,
                                      PrepareVulkanMatrix);
}

rund::AccelCheck ExecuteVulkanTransform(const rund::AccelDevice &pick,
                                        const rund::kernel::TransformDesc &desc,
                                        const rund::kernel::TransformPlan &plan,
                                        const TransformBinds &bindings) {
  return ExecutePreparedVulkanNumeric(pick, desc, plan, bindings,
                                      PrepareVulkanTransform);
}

rund::AccelCheck ExecuteVulkanFactor(const rund::AccelDevice &pick,
                                     const rund::kernel::FactorDesc &desc,
                                     const rund::kernel::FactorPlan &plan,
                                     const FactorBinds &bindings) {
  return ExecutePreparedVulkanNumeric(pick, desc, plan, bindings,
                                      PrepareVulkanFactor);
}

rund::AccelCheck ExecuteVulkanSolve(const rund::AccelDevice &pick,
                                    const rund::kernel::SolveDesc &desc,
                                    const rund::kernel::SolvePlan &plan,
                                    const SolveBinds &bindings) {
  return ExecutePreparedVulkanNumeric(pick, desc, plan, bindings,
                                      PrepareVulkanSolve);
}

rund::AccelCheck ExecuteVulkanSpectrum(const rund::AccelDevice &pick,
                                       const rund::kernel::SpectrumDesc &desc,
                                       const rund::kernel::SpectrumPlan &plan,
                                       const SpectrumBinds &bindings) {
  return ExecutePreparedVulkanNumeric(pick, desc, plan, bindings,
                                      PrepareVulkanSpectrum);
}

} // namespace rund::node::accel::detail
