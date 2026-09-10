#include "src/accel/backend/ops/table.hpp"
#include "src/accel/context/internal/execution.hpp"
#include "src/accel/kernel/memory.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"

#include "src/accel/metal/kernel/pipeline/identity/index.hpp"

#include <cstdint>
#include <limits>
#include <memory>

#include "cache.hpp"
#include "hooks.hpp"

namespace node_accel_contract {

namespace {

[[nodiscard]] bool MatchPreparedInteger(const void *const prepared,
                                        const void *const probe) noexcept {
  return prepared != nullptr && probe != nullptr &&
         *static_cast<const int *>(prepared) ==
             *static_cast<const int *>(probe);
}

[[nodiscard]] rund::node::accel::detail::PreparedMemory
ObservePreparedInteger(const void *const prepared) noexcept {
  using rund::node::accel::detail::PreparedMemory;
  return prepared == nullptr ? PreparedMemory{}
                             : PreparedMemory{.current = sizeof(int),
                                              .peak = sizeof(int),
                                              .cumulative = sizeof(int),
                                              .budget = sizeof(int)};
}

} // namespace

[[nodiscard]] bool PreparedTemplateRegistryIsColdAndCollisionSafe() {
  using namespace rund::node::accel::detail;
  const BackendOps ops{
      .api = rund::AccelApi::Metal,
      .observe_pipeline_template = ObservePreparedInteger,
  };
  PreparedKernelTemplateRegistry registry{};
  const PreparedKernelPipelineReservation invalid =
      PlanPreparedKernelPipelineLimit({}, {}, {}, registry);
  if (invalid.ok || registry.owner != nullptr || registry.limit.ok) {
    return false;
  }
  if (!BindPreparedKernelTemplateRegistry(rund::AccelApi::Metal, 41u, registry)
           .ok ||
      registry.owner == nullptr) {
    return false;
  }

  KernelExecutionStep authority{};
  int first_probe = 7;
  std::shared_ptr<void> first = std::make_shared<int>(first_probe);
  if (!PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                     MatchPreparedInteger, &first_probe, first)
           .ok) {
    return false;
  }
  const std::shared_ptr<void> found = FindPreparedKernelTemplate(
      registry, &authority, 11u, 13u, MatchPreparedInteger, &first_probe);
  if (found != first) {
    return false;
  }

  // Equal hashes are only a partition. Semantic mismatch must publish a
  // second immutable owner instead of aliasing the collision.
  int second_probe = 9;
  std::shared_ptr<void> second = std::make_shared<int>(second_probe);
  if (!PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                     MatchPreparedInteger, &second_probe,
                                     second)
           .ok ||
      second == first ||
      FindPreparedKernelTemplate(registry, &authority, 11u, 13u,
                                 MatchPreparedInteger,
                                 &second_probe) != second) {
    return false;
  }

  std::shared_ptr<void> duplicate = std::make_shared<int>(first_probe);
  return PublishPreparedKernelTemplate(registry, &authority, 11u, 13u, ops,
                                       MatchPreparedInteger, &first_probe,
                                       duplicate)
             .ok &&
         duplicate == first;
}

[[nodiscard]] bool RecurrenceRouteCopiesDoNotCloneTemplates() {
  using namespace rund::node::accel::detail;
  const PreparedMapRecurrenceReservation one{
      .route_host_bytes = 11u,
      .route_native_bytes = 13u,
      .template_host_bytes = 17u,
      .template_native_bytes = 19u,
      .template_source_bytes = 23u,
      .source_transient_bytes = 29u,
      .group_count = 3u,
      .history_group_count = 1u,
      .template_count = 2u,
      .terminal_template_group_capacity = 6u,
      .history_template_group_capacity = 2u,
      .route_step_count = 3u,
      .template_step_count = 2u,
      .descriptor_set_count = 31u,
      .descriptor_count = 37u,
      .route_native_allocation_count = 41u,
      .template_native_allocation_count = 43u,
  };
  PreparedMapRecurrenceReservation two{};
  return ScalePreparedMapRecurrenceRoutesForContract(one, 2u, two) &&
         two.route_host_bytes == 22u && two.route_native_bytes == 26u &&
         two.group_count == 6u && two.history_group_count == 2u &&
         two.route_step_count == 6u && two.descriptor_set_count == 0u &&
         two.descriptor_count == 0u &&
         two.route_native_allocation_count == 82u &&
         two.template_host_bytes == 0u && two.template_native_bytes == 0u &&
         two.template_source_bytes == 0u && two.source_transient_bytes == 0u &&
         two.template_count == 0u &&
         two.terminal_template_group_capacity == 0u &&
         two.history_template_group_capacity == 0u &&
         two.template_step_count == 0u &&
         two.template_native_allocation_count == 0u &&
         !ScalePreparedMapRecurrenceRoutesForContract(one, 0u, two);
}

[[nodiscard]] bool MetalPointerIdentityIndexIsExactAndOneShot() {
  using namespace rund::node::accel::detail;
  const MetalPointerIdentityIndexLayout empty =
      PlanMetalPointerIdentityIndex(0u);
  const MetalPointerIdentityIndexLayout one = PlanMetalPointerIdentityIndex(1u);
  const MetalPointerIdentityIndexLayout three =
      PlanMetalPointerIdentityIndex(3u);
  const MetalPointerIdentityIndexLayout eight =
      PlanMetalPointerIdentityIndex(8u);
  const MetalPointerIdentityIndexLayout overflow =
      PlanMetalPointerIdentityIndex(
          std::numeric_limits<std::uint64_t>::max() / 2u + 1u);
  if (!empty.ok || empty.slot_count != 0u || empty.byte_count != 0u ||
      !one.ok || one.slot_count != 2u ||
      one.byte_count != 2u * sizeof(const void *) || !three.ok ||
      three.slot_count != 8u || three.byte_count != 8u * sizeof(const void *) ||
      !eight.ok || eight.slot_count != 16u ||
      eight.byte_count != 16u * sizeof(const void *) || overflow.ok) {
    return false;
  }

  int first = 0;
  int second = 0;
  int third = 0;
  MetalPointerIdentityIndex index{3u};
  bool inserted = false;
  if (!index.ready() || !index.insert(&first, inserted) || !inserted ||
      !index.insert(&first, inserted) || inserted ||
      !index.insert(&second, inserted) || !inserted) {
    return false;
  }
  index.clear();
  return index.insert(&first, inserted) && inserted &&
         index.insert(&second, inserted) && inserted &&
         index.insert(&third, inserted) && inserted;
}

} // namespace node_accel_contract
