#include "local.hpp"

#include "../../allocation.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace rund_node_test_pipeline_residency {

int CheckGraphPlanner() {
  using rund::compute::detail::Type;
  using namespace rund::compute::detail::residency;
  const TiledGraphPlanInput recurrent_input{
      .page_count = 7u,
      .requested_frames = 2u,
      .max_frames = 2u,
      .prefetch_distance = 2u,
      .resources = {TiledGraphResourceInput{
                        .resource = 1u,
                        .type = Type::U64,
                        .page_bytes = 64u,
                        .logical_bytes = 7u * 64u,
                        .kind = GraphResourceKind::ExternalInput,
                        .persistence = ResourcePersistence::Backing},
                    TiledGraphResourceInput{.resource = 3u,
                                            .type = Type::U64,
                                            .page_bytes = 64u,
                                            .logical_bytes = 6u * 64u + 40u,
                                            .kind = GraphResourceKind::Internal,
                                            .persistence =
                                                ResourcePersistence::Transient},
                    TiledGraphResourceInput{
                        .resource = 4u,
                        .type = Type::U64,
                        .page_bytes = 8u,
                        .logical_bytes = 7u * 8u,
                        .kind = GraphResourceKind::ExternalOutput,
                        .persistence = ResourcePersistence::Transient}},
      .stages = {TiledGraphStageInput{
                     .node = 0u,
                     .ports = {{.resource = 1u, .access = Access::Read},
                               {.resource = 3u, .access = Access::Write}}},
                 TiledGraphStageInput{
                     .node = 1u,
                     .ports = {{.resource = 3u, .access = Access::Read},
                               {.resource = 4u, .access = Access::Write}}}},
  };
  // Preparation borrows the request but the sealed plan owns its remaps.
  // Resource order is canonical; caller order and caller storage stay intact.
  auto remapped_input = recurrent_input;
  remapped_input.resources[0].remaps = {
      {.source_local = 0u, .target_local = 1u},
      {.source_local = 1u, .target_local = 0u}};
  const auto canonical = PlanResidency(remapped_input);
  std::reverse(remapped_input.resources.begin(), remapped_input.resources.end());
  const auto original_resources = remapped_input.resources;
  const auto original_stages = remapped_input.stages;
  const auto remapped = PlanResidency(remapped_input);
  if (!canonical || !remapped ||
      canonical.plan.identity() != remapped.plan.identity() ||
      remapped_input.resources != original_resources ||
      remapped_input.stages != original_stages) {
    return 26;
  }
  remapped_input.resources.clear();
  remapped_input.stages.clear();
  const auto &owned = remapped.plan.tiled_graph();
  const std::array<std::uint64_t, 3u> remapped_bytes{7u * 64u,
                                                  6u * 64u + 40u, 7u * 8u};
  bool warm_valid = true;
  node_compute_allocation::Start();
  for (unsigned repeat = 0u; repeat < 8u; ++repeat) {
    TiledGraphInvocation active{};
    const auto *resource = owned.resource(1u);
    warm_valid = warm_valid && owned.active(7u, remapped_bytes, active) &&
                 resource != nullptr && resource->remaps.size() == 2u &&
                 resource->remaps[0].target_local == 0u &&
                 resource->remaps[0].source_local == 1u &&
                 owned.resource(2u) == nullptr;
  }
  node_compute_allocation::Stop();
  if (!warm_valid || node_compute_allocation::Count() != 0u) {
    return 27;
  }
  auto recurrent = PlanResidency(recurrent_input);
  if (!recurrent || recurrent.plan.streamed() ||
      !recurrent.plan.graph_tiled() || recurrent.plan.page_bytes() != 136u ||
      recurrent.plan.frame_capacity() != 2u ||
      recurrent.plan.prefetch_distance() != 2u || !recurrent.plan.identity()) {
    return 8;
  }
  const TiledGraphPlan &stream = recurrent.plan.tiled_graph();
  TiledGraphInvocation invocation{};
  const std::array<std::uint64_t, 3u> invocation_bytes{7u * 64u, 6u * 64u + 40u,
                                                       7u * 8u};
  PageRun tail{};
  std::uint64_t first_use = NeverUse;
  if (stream.page_count() != 7u || stream.frame_capacity() != 2u ||
      stream.batch_count() != 4u || stream.stage_count() != 2u ||
      stream.epoch_count() != 8u || stream.resources().size() != 3u ||
      stream.stages().size() != 2u ||
      stream.stages()[0].domain != StageDomain::Tile ||
      stream.stages()[0].active_count_input != 1u ||
      !stream.active(7u, invocation_bytes, invocation) ||
      !invocation.batch(3u, tail) || tail.first_page != 6u ||
      tail.page_count != 1u || tail.pin != PinInterval{6u, 7u} ||
      tail.prefetch_epoch != 2u || tail.ready_epoch != 6u ||
      !invocation.first_use(1u, 0u, first_use) || first_use != 0u ||
      !invocation.first_use(1u, 3u, first_use) || first_use != 2u ||
      !invocation.first_use(1u, 6u, first_use) || first_use != 6u ||
      invocation.first_use(1u, 7u, first_use) ||
      invocation.first_use(3u, 0u, first_use)) {
    return 9;
  }
  std::array<PageUse, 2u> uses{};
  std::array<TiledGraphDependency, TiledGraphDependencyCapacity> dependencies{};
  std::size_t dependency_count = 0u;
  Epoch recurrent_epoch{};
  if (!invocation.project(3u, 0u, uses, recurrent_epoch) ||
      recurrent_epoch.node != 0u || recurrent_epoch.tile != 3u ||
      recurrent_epoch.use_count != 2u ||
      uses[0].key != PageKey{.resource = 1u, .page = 6u} ||
      uses[0].access != Access::Read || uses[0].next_use != 14u ||
      uses[0].pin != PinInterval{6u, 6u} || uses[0].prefetch_epoch != 2u ||
      uses[0].ready_epoch != 6u ||
      uses[1].key != PageKey{.resource = 3u, .page = 6u} ||
      uses[1].access != Access::Write ||
      uses[1].dirty != DirtyRange{.bytes = 40u} || uses[1].next_use != 7u ||
      uses[1].pin != PinInterval{6u, 7u} ||
      !invocation.project(3u, 1u, uses, recurrent_epoch) ||
      recurrent_epoch.node != 1u || recurrent_epoch.tile != 3u ||
      uses[0].key != PageKey{.resource = 3u, .page = 6u} ||
      uses[0].next_use != NeverUse || uses[0].pin != PinInterval{6u, 7u} ||
      uses[1].key != PageKey{.resource = 4u, .page = 6u} ||
      uses[1].dirty != DirtyRange{.bytes = 8u} ||
      uses[1].pin != PinInterval{7u, 7u}) {
    return 10;
  }
  if (!invocation.predecessors(0u, 0u, dependencies, dependency_count) ||
      dependency_count != 0u ||
      !invocation.predecessors(0u, 1u, dependencies, dependency_count) ||
      dependency_count != 1u ||
      dependencies[0] !=
          TiledGraphDependency{.ordinal = 0u, .batch = 0u, .stage = 0u} ||
      !invocation.predecessors(1u, 0u, dependencies, dependency_count) ||
      dependency_count != 2u ||
      dependencies[0] !=
          TiledGraphDependency{.ordinal = 0u, .batch = 0u, .stage = 0u} ||
      dependencies[1] !=
          TiledGraphDependency{.ordinal = 1u, .batch = 0u, .stage = 1u}) {
    return 22;
  }
  if (!invocation.predecessors(1u, 1u, dependencies, dependency_count) ||
      dependency_count != 2u ||
      dependencies[0] !=
          TiledGraphDependency{
              .ordinal = 1u,
              .batch = 0u,
              .stage = 1u,
              .phase = TiledGraphDependencyPhase::ReleaseComplete} ||
      dependencies[1] !=
          TiledGraphDependency{
              .ordinal = 2u,
              .batch = 1u,
              .stage = 0u,
              .phase = TiledGraphDependencyPhase::DispatchComplete}) {
    return 26;
  }
  auto changed_recurrent = recurrent_input;
  changed_recurrent.resources[1].logical_bytes -= sizeof(std::uint64_t);
  const auto changed_recurrent_plan = PlanResidency(changed_recurrent);
  if (!changed_recurrent_plan ||
      changed_recurrent_plan.plan.identity() == recurrent.plan.identity()) {
    return 11;
  }
  auto invalid_recurrent = recurrent_input;
  invalid_recurrent.resources[1].logical_bytes = 6u * 64u;
  if (PlanResidency(invalid_recurrent).failure != Failure::Invalid) {
    return 12;
  }

  const TiledGraphPlanInput colored_input{
      .page_count = 3u,
      .requested_frames = 2u,
      .max_frames = 2u,
      .prefetch_distance = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 5u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 6u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 7u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 3u * 64u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 0u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 2u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}},
                 {.node = 1u,
                  .ports = {{.resource = 3u, .access = Access::Read},
                            {.resource = 5u, .access = Access::Write}}},
                 {.node = 2u,
                  .ports = {{.resource = 5u, .access = Access::Read},
                            {.resource = 1u, .access = Access::Read},
                            {.resource = 6u, .access = Access::Write}}},
                 {.node = 3u,
                  .ports = {{.resource = 6u, .access = Access::Read},
                            {.resource = 7u, .access = Access::Write}}}},
  };
  const PlanResult colored = PlanResidency(colored_input);
  if (!colored || !colored.plan.graph_tiled()) {
    return 13;
  }
  const TiledGraphPlan &colored_plan = colored.plan.tiled_graph();
  const TiledGraphResource *const input_a = colored_plan.resource(1u);
  const TiledGraphResource *const input_b = colored_plan.resource(2u);
  const TiledGraphResource *const intermediate_a = colored_plan.resource(3u);
  const TiledGraphResource *const intermediate_b = colored_plan.resource(5u);
  const TiledGraphResource *const intermediate_c = colored_plan.resource(6u);
  const TiledGraphResource *const colored_output = colored_plan.resource(7u);
  if (input_a == nullptr || input_b == nullptr || intermediate_a == nullptr ||
      intermediate_b == nullptr || intermediate_c == nullptr ||
      colored_output == nullptr || colored.plan.page_bytes() != 5u * 64u ||
      colored_plan.physical_classes().size() != 5u ||
      input_a->color == input_b->color ||
      intermediate_a->color == intermediate_b->color ||
      intermediate_b->color == intermediate_c->color ||
      intermediate_a->color != intermediate_c->color ||
      intermediate_a->physical_id != intermediate_c->physical_id ||
      intermediate_a->physical_id == intermediate_b->physical_id ||
      intermediate_a->first_stage != 0u || intermediate_a->last_stage != 1u ||
      intermediate_a->producer_stage != 0u ||
      intermediate_a->role != GraphResourceRole::Intermediate ||
      colored_output->role != GraphResourceRole::Output ||
      colored_plan.stages()[0].ports[1].program_port != 1u ||
      colored_plan.stages()[0].ports[2].next_stage != 1u) {
    return 14;
  }
  auto later_input = colored_input;
  later_input.resources.erase(later_input.resources.begin() + 3u,
                              later_input.resources.begin() + 5u);
  later_input.stages = {
      TiledGraphStageInput{
          .node = 0u,
          .ports = {{.resource = 1u, .access = Access::Read},
                    {.resource = 3u, .access = Access::Write}}},
      TiledGraphStageInput{
          .node = 1u,
          .ports = {{.resource = 3u, .access = Access::Read},
                    {.resource = 2u, .access = Access::Read},
                    {.resource = 7u, .access = Access::Write}}},
  };
  const PlanResult later = PlanResidency(later_input);
  const TiledGraphResource *const later_first =
      later ? later.plan.tiled_graph().resource(1u) : nullptr;
  const TiledGraphResource *const later_second =
      later ? later.plan.tiled_graph().resource(2u) : nullptr;
  if (!later || later_first == nullptr || later_second == nullptr ||
      later_first->last_stage != 0u || later_second->first_stage != 1u ||
      later_first->color == later_second->color ||
      later_first->physical_id == later_second->physical_id ||
      later.plan.tiled_graph().physical_classes().size() != 4u) {
    return 27;
  }
  TiledGraphInvocation colored_invocation{};
  const std::array<std::uint64_t, 6u> colored_bytes{
      3u * 64u, 3u * 64u, 3u * 64u, 3u * 64u, 3u * 64u, 3u * 64u};
  std::array<PageUse, 6u> colored_uses{};
  Epoch colored_epoch{};
  if (!colored_plan.active(3u, colored_bytes, colored_invocation) ||
      !colored_invocation.project(0u, 0u, colored_uses, colored_epoch) ||
      colored_epoch.use_count != colored_uses.size() ||
      colored_uses[0].key != PageKey{.resource = 1u, .page = 0u} ||
      colored_uses[0].next_use != 2u ||
      colored_uses[0].pin != PinInterval{0u, 0u} ||
      colored_uses[2].key != PageKey{.resource = 2u, .page = 0u} ||
      colored_uses[4].key != PageKey{.resource = 3u, .page = 0u}) {
    return 15;
  }
  if (!colored_invocation.predecessors(0u, 1u, dependencies,
                                       dependency_count) ||
      dependency_count != 1u || dependencies[0].stage != 0u ||
      !colored_invocation.predecessors(0u, 2u, dependencies,
                                       dependency_count) ||
      dependency_count != 1u || dependencies[0].stage != 1u ||
      !colored_invocation.predecessors(0u, 3u, dependencies,
                                       dependency_count) ||
      dependency_count != 1u || dependencies[0].stage != 2u) {
    return 23;
  }
  if (!colored_invocation.predecessors(1u, 0u, dependencies,
                                       dependency_count) ||
      dependency_count != 3u || dependencies[0].stage != 0u ||
      dependencies[1].stage != 2u || dependencies[2].stage != 3u) {
    return 24;
  }
  if (!colored_invocation.project(0u, 2u, colored_uses, colored_epoch) ||
      colored_epoch.ordinal != 2u ||
      colored_uses[2].key != PageKey{.resource = 1u, .page = 0u} ||
      colored_uses[2].next_use != colored_invocation.epoch_count() + 2u ||
      colored_uses[2].pin != PinInterval{2u, 2u}) {
    return 21;
  }
  auto misaligned_bytes = colored_bytes;
  --misaligned_bytes[2];
  if (colored_plan.active(3u, misaligned_bytes, colored_invocation)) {
    return 16;
  }
  auto changed_port_order = colored_input;
  std::swap(changed_port_order.stages[0].ports[0],
            changed_port_order.stages[0].ports[1]);
  const PlanResult changed_port_plan = PlanResidency(changed_port_order);
  if (!changed_port_plan ||
      changed_port_plan.plan.identity() == colored.plan.identity()) {
    return 17;
  }
  auto invalid_transient = colored_input;
  invalid_transient.stages[0].ports[2].access = Access::ReadWrite;
  if (PlanResidency(invalid_transient).failure != Failure::Invalid) {
    return 18;
  }
  auto invalid_domain = colored_input;
  invalid_domain.stages.back().domain = StageDomain::Terminal;
  if (PlanResidency(invalid_domain).failure != Failure::Invalid) {
    return 19;
  }

  // Graph order is not a dependency. The second branch has different
  // physical compatibility classes and can advance independently; each join
  // waits only its sealed transient producer.
  const TiledGraphPlanInput branches_input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 128u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 4u,
                     .type = Type::U64,
                     .page_bytes = 128u,
                     .logical_bytes = 128u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 5u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 0u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}},
                 {.node = 1u,
                  .ports = {{.resource = 2u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Write}}},
                 {.node = 2u,
                  .ports = {{.resource = 3u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Read},
                            {.resource = 5u, .access = Access::Write}}}},
  };
  PlanResult branches = PlanResidency(branches_input);
  TiledGraphInvocation branch_invocation{};
  const std::array<std::uint64_t, 5u> branch_bytes{64u, 128u, 64u, 128u, 64u};
  if (!branches ||
      !branches.plan.tiled_graph().active(1u, branch_bytes,
                                          branch_invocation) ||
      !branch_invocation.predecessors(0u, 0u, dependencies, dependency_count) ||
      dependency_count != 0u ||
      !branch_invocation.predecessors(0u, 1u, dependencies, dependency_count) ||
      dependency_count != 0u ||
      !branch_invocation.predecessors(0u, 2u, dependencies, dependency_count) ||
      dependency_count != 2u || dependencies[0].stage != 0u ||
      dependencies[1].stage != 1u) {
    return 25;
  }
  invalid_recurrent = recurrent_input;
  invalid_recurrent.requested_frames = 0u;
  invalid_recurrent.max_frames = 0u;
  return PlanResidency(invalid_recurrent).failure == Failure::Infeasible ? 0
                                                                         : 20;
}

} // namespace rund_node_test_pipeline_residency
