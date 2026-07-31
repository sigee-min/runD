#include "../resource.hpp"

#include "../../adapter.hpp"
#include "../../command/run.hpp"

namespace rund::node::accel::detail {
namespace {

rund::AccelCheck
ExecutePreparedMetalNumeric(const rund::AccelDevice &pick,
                            const std::shared_ptr<void> &prepared) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  @autoreleasepool {
    MetalAdapter *const adapter = MetalAdapterFromPick(pick);
    if (adapter == nullptr || prepared == nullptr) {
      return rund::AccelCheck{false, "accel_metal_unavailable"};
    }
    CommandRun command{};
    rund::AccelCheck check = OpenCommand(*adapter, command);
    if (check.ok) {
      check = EncodeMetalNumeric(*adapter, prepared,
                                 (__bridge void *)command.encoder);
      check = FinishCommand(*adapter, command, check);
    }
    return check.ok ? FinishMetalNumeric(*adapter, prepared) : check;
  }
#else
  (void)pick;
  (void)prepared;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace

rund::AccelCheck ExecuteMetalMatrix(const rund::AccelDevice &pick,
                                    const rund::kernel::MatrixDesc &desc,
                                    const rund::kernel::MatrixPlan &plan,
                                    const MatrixBinds &bindings) {
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check =
      PrepareMetalMatrix(pick, desc, plan, bindings, prepared);
  return check.ok ? ExecutePreparedMetalNumeric(pick, prepared) : check;
}

rund::AccelCheck ExecuteMetalTransform(const rund::AccelDevice &pick,
                                       const rund::kernel::TransformDesc &desc,
                                       const rund::kernel::TransformPlan &plan,
                                       const TransformBinds &bindings) {
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check =
      PrepareMetalTransform(pick, desc, plan, bindings, prepared);
  return check.ok ? ExecutePreparedMetalNumeric(pick, prepared) : check;
}

rund::AccelCheck ExecuteMetalFactor(const rund::AccelDevice &pick,
                                    const rund::kernel::FactorDesc &desc,
                                    const rund::kernel::FactorPlan &plan,
                                    const FactorBinds &bindings) {
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check =
      PrepareMetalFactor(pick, desc, plan, bindings, prepared);
  return check.ok ? ExecutePreparedMetalNumeric(pick, prepared) : check;
}

rund::AccelCheck ExecuteMetalSolve(const rund::AccelDevice &pick,
                                   const rund::kernel::SolveDesc &desc,
                                   const rund::kernel::SolvePlan &plan,
                                   const SolveBinds &bindings) {
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check =
      PrepareMetalSolve(pick, desc, plan, bindings, prepared);
  return check.ok ? ExecutePreparedMetalNumeric(pick, prepared) : check;
}

rund::AccelCheck ExecuteMetalSpectrum(const rund::AccelDevice &pick,
                                      const rund::kernel::SpectrumDesc &desc,
                                      const rund::kernel::SpectrumPlan &plan,
                                      const SpectrumBinds &bindings) {
  std::shared_ptr<void> prepared;
  const rund::AccelCheck check =
      PrepareMetalSpectrum(pick, desc, plan, bindings, prepared);
  return check.ok ? ExecutePreparedMetalNumeric(pick, prepared) : check;
}

} // namespace rund::node::accel::detail
