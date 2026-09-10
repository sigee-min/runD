#include "local.hpp"

#include "src/accel/kernel/reset/proof.hpp"
#include "src/accel/vulkan/kernel/reset.hpp"

namespace node_accel_contract::reset_contract {
namespace {

using rund::node::accel::detail::KernelPreparationMode;
using rund::node::accel::detail::PlanVulkanResetExecution;
using rund::node::accel::detail::reset::Prove;
using rund::node::accel::detail::reset::Spec;

} // namespace

bool CheckVulkanExecution() {
  constexpr std::uint64_t Count = 513u;
  constexpr std::uint64_t Window = 256u;
  const Range dense =
      Prove(Spec{.count = Count, .stride = 4u, .element = 4u}, Count * 4u);
  const Range strided = Prove(Spec{.count = Count, .stride = 8u, .element = 4u},
                              (Count - 1u) * 8u + 4u);
  if (!Accepted(dense) || !Accepted(strided)) {
    return false;
  }
  const auto standalone_dense = PlanVulkanResetExecution(
      dense, KernelPreparationMode::Standalone, Window);
  const auto captured_dense = PlanVulkanResetExecution(
      dense, KernelPreparationMode::PipelinePrivate, Window);
  const auto standalone_strided = PlanVulkanResetExecution(
      strided, KernelPreparationMode::Standalone, Window);
  const auto dense_without_dispatch =
      PlanVulkanResetExecution(dense, KernelPreparationMode::Standalone, 0u);
  const auto captured_without_dispatch = PlanVulkanResetExecution(
      dense, KernelPreparationMode::PipelinePrivate, 0u);
  return standalone_dense.ok && !standalone_dense.shader &&
         standalone_dense.commands == 1u && captured_dense.ok &&
         captured_dense.shader && captured_dense.commands == 3u &&
         standalone_strided.ok && standalone_strided.shader &&
         standalone_strided.commands == 3u && dense_without_dispatch.ok &&
         !dense_without_dispatch.shader &&
         dense_without_dispatch.commands == 1u && !captured_without_dispatch.ok;
}

} // namespace node_accel_contract::reset_contract
