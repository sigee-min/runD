#include "internal.hpp"

#include "../../../../pipeline/guard.hpp"
#include "../../../../runtime/map/api.hpp"
#include "../../../../runtime/map/resources.hpp"

#include "../../../../../kernel/recurrence/plan.hpp"
#include "../../../../../kernel/recurrence/source.hpp"
#include "../../../../../kernel/step/map/stride.hpp"

#include <span>

namespace rund::node::accel::detail::metal_pipeline_admit_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] std::span<const std::uint64_t>
RecurrenceHistoryPitches(const MapRecurrence &recurrence) noexcept {
  return recurrence.history == nullptr ? std::span<const std::uint64_t>{}
                                       : recurrence.history->pitches();
}

struct MetalRecurrenceTemplateProbe final {
  const MapRecurrencePreparationPlan *plan{};
  const rund::kernel::BindingSet *bindings{};
  MetalAdapter *adapter{};
  bool history{};
};

[[nodiscard]] bool
MatchMetalRecurrenceTemplate(const void *const prepared,
                             const void *const raw_probe) noexcept {
  if (prepared == nullptr || raw_probe == nullptr ||
      MetalKernelTemplateKindOf(prepared) !=
          MetalKernelTemplateKind::MapRecurrence) {
    return false;
  }
  const auto *const cached =
      static_cast<const MetalMapRecurrenceTemplate *>(prepared);
  const auto *const probe =
      static_cast<const MetalRecurrenceTemplateProbe *>(raw_probe);
  if (cached->signature == nullptr || cached->prepared == nullptr ||
      cached->prepared->adapter != probe->adapter || probe->plan == nullptr ||
      probe->bindings == nullptr || probe->adapter == nullptr ||
      cached->history != probe->history) {
    return false;
  }
  const MapRecurrencePreparationPlan cached_plan = PlanMapRecurrencePreparation(
      *cached->signature, 1u, cached->history ? 1u : 0u);
  return SameMapRecurrenceTemplate(cached_plan, *probe->plan, probe->history) &&
         MetalMapTemplateMatches(*cached->prepared, *probe->adapter,
                                 probe->plan->plan, *probe->bindings);
}

inline constexpr std::uint64_t MetalRecurrenceVariantHi = 0x6d6574616c2e7265ull;
inline constexpr std::uint64_t MetalTerminalRecurrenceVariantLo =
    0x6375722e7465726dull;
inline constexpr std::uint64_t MetalHistoryRecurrenceVariantLo =
    0x6375722e68697374ull;

[[nodiscard]] rund::AccelCheck AcquireMetalRecurrenceTemplate(
    PreparedKernelTemplateRegistry &registry, MetalAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence,
    std::shared_ptr<const MetalMapTemplateResources> &prepared) {
  prepared.reset();
  const std::span<const std::uint64_t> history_pitch_bytes =
      RecurrenceHistoryPitches(recurrence);
  const bool history = !history_pitch_bytes.empty();
  const MapRecurrencePreparationPlan normalized =
      PlanMapRecurrencePreparation(signature, 1u, history ? 1u : 0u);
  const MapRecurrenceSourcePlan &source_plan =
      history ? normalized.history_source : normalized.terminal_source;
  if (registry.owner == nullptr || recurrence.first == nullptr ||
      recurrence.first->step == nullptr || signature.steps == nullptr ||
      signature.ops == nullptr || signature.step_count != 1u ||
      signature.steps[0u].step != recurrence.first->step ||
      !normalized.eligible() || normalized.group_count != 1u ||
      normalized.history_group_count != (history ? 1u : 0u) ||
      normalized.binding_alignment != 1u ||
      normalized.authority != recurrence.first->step ||
      normalized.canonical_artifact == nullptr || !source_plan.ok ||
      source_plan.history != history ||
      (history && history_pitch_bytes.size() != normalized.output_count)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::uint64_t variant_lo = history ? MetalHistoryRecurrenceVariantLo
                                           : MetalTerminalRecurrenceVariantLo;
  const MetalRecurrenceTemplateProbe probe{
      .plan = &normalized,
      .bindings = &recurrence.bindings,
      .adapter = &adapter,
      .history = history,
  };
  std::shared_ptr<void> found = FindPreparedKernelTemplate(
      registry, normalized.authority, MetalRecurrenceVariantHi, variant_lo,
      MatchMetalRecurrenceTemplate, &probe);
  if (found != nullptr) {
    const auto cached =
        std::static_pointer_cast<MetalMapRecurrenceTemplate>(found);
    if (cached == nullptr || cached->prepared == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    prepared = cached->prepared;
    return rund::AccelCheck{true, "ok"};
  }

  // A miss owns exactly one transformed source. PrepareMetalMapTemplate moves
  // the specialized/guarded text into the adapter cache; this local artifact
  // dies before any route resource is allocated. Reserve the final guarded
  // upper during initial recurrence emission so every later edit is in-place.
  std::uint64_t specialized_upper = 0u;
  std::uint64_t final_source_upper = 0u;
  if (recurrence.first->control.active() ||
      !normalized.canonical_artifact->metadata.read_routes.empty() ||
      !MapSpecializedSourceUpperBytes(source_plan.exact_source_bytes,
                                      source_plan.source_upper_bytes,
                                      normalized.plan, specialized_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(specialized_upper, 1u, true,
                                            final_source_upper)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  rund::kernel::LoweringArtifact artifact{};
  if (!MaterializeMapRecurrenceArtifact(
          *normalized.canonical_artifact, source_plan, normalized.input_count,
          normalized.output_count, history_pitch_bytes, artifact,
          final_source_upper)) {
    return rund::AccelCheck{false,
                            "compute_pipeline_recurrence_source_invalid"};
  }
  std::shared_ptr<const MetalMapTemplateResources> native;
  const rund::AccelCheck ready = PrepareMetalMapOwnedTemplate(
      pick, normalized.plan, std::move(artifact), recurrence.windows,
      recurrence.window_count, recurrence.bindings, recurrence.first->control,
      native);
  if (!ready.ok) {
    return ready;
  }

  auto owner = std::make_shared<MetalMapRecurrenceTemplate>();
  owner->signature = &signature;
  owner->history = history;
  owner->prepared = std::move(native);
  std::shared_ptr<void> published = owner;
  const rund::AccelCheck stored = PublishPreparedKernelTemplate(
      registry, normalized.authority, MetalRecurrenceVariantHi, variant_lo,
      *signature.ops, MatchMetalRecurrenceTemplate, &probe, published);
  if (!stored.ok || published == nullptr ||
      !MatchMetalRecurrenceTemplate(published.get(), &probe)) {
    return stored.ok ? rund::AccelCheck{false, "accel_kernel_template_invalid"}
                     : stored;
  }
  const auto cached =
      std::static_pointer_cast<MetalMapRecurrenceTemplate>(published);
  if (cached == nullptr || cached->prepared == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  prepared = cached->prepared;
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck PrepareMetalRecurrenceRoute(
    PreparedKernelTemplateRegistry &registry, MetalAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence, const BoundControl &control,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  std::shared_ptr<const MetalMapTemplateResources> prepared;
  const rund::AccelCheck acquired = AcquireMetalRecurrenceTemplate(
      registry, adapter, pick, signature, recurrence, prepared);
  return acquired.ok ? PrepareMetalMapProvedRoute(
                           pick, recurrence.plan, recurrence.windows,
                           recurrence.window_count, recurrence.bindings,
                           control, std::move(prepared), resources, iterations)
                     : acquired;
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_admit_internal
