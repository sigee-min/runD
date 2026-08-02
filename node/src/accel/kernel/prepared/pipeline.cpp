#include "../prepared.hpp"

#include "../../context/internal/support.hpp"
#include "../backend/pipeline_failure.hpp"
#include "../backend/template_plan.hpp"
#include "../recurrence.hpp"
#include "../recurrence/plan.hpp"
#include "evidence.hpp"
#include "model.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

// Narrow contract seam for the allocation-free route projection. Production
// planning uses the same implementation below for public and runtime paths.
[[nodiscard]] bool AccumulatePreparedKernelRouteProjectionForContract(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    std::uint64_t compact_entry_count, std::uint64_t occurrence_count,
    std::uint64_t window_count) noexcept;
[[nodiscard]] bool BackendTemplateRouteDemandForContract(
    std::uint64_t owner_count, std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept;
[[nodiscard]] bool PlanBackendTemplateRouteDemandsForContract(
    std::span<const PreparedKernelRun *const> runs, std::uint32_t route_copies,
    std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept;
[[nodiscard]] bool BackendPreparationCursorLifecycleForContract(
    BackendRun &run, BackendTemplateRouteDemand demand) noexcept;
[[nodiscard]] bool ScalePreparedMapRecurrenceRoutesForContract(
    const PreparedMapRecurrenceReservation &reservation, std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept;

namespace {

inline constexpr std::uint64_t kTemplateRegistryMagic = 0x72756e442e74706cull;

struct PreparedKernelTemplateEntry final {
  const KernelExecutionStep *authority{};
  std::uint64_t variant_hi{};
  std::uint64_t variant_lo{};
  const BackendOps *ops{};
  std::shared_ptr<void> prepared{};
};

enum class PreparedKernelTemplateChargeKind : std::uint8_t {
  Program,
  RecurrenceTerminal,
  RecurrenceHistory,
};

struct PreparedKernelTemplateCharge final {
  std::shared_ptr<prepared::RunState> probe{};
  const BackendOps *ops{};
  PreparedKernelTemplateChargeKind kind{
      PreparedKernelTemplateChargeKind::Program};
};

struct PreparedKernelTemplateRegistryState final {
  // Cold preparation holds this recursively across reservation, template
  // publication and backend finalization. Recursive acquisition is required
  // because Find/Publish are called by the same preparation thread while the
  // transaction excludes a competing primary/alternate stream.
  std::recursive_mutex mutex{};
  std::vector<PreparedKernelTemplateEntry> entries{};
  std::vector<PreparedKernelTemplateCharge> template_charges{};
  PreparedKernelPipelineReservation consumed{};
  std::uint64_t magic{kTemplateRegistryMagic};
  std::uint64_t context_id{};
  rund::AccelApi api{rund::AccelApi::Auto};
};

class PipelineBudgetTransaction final {
public:
  PipelineBudgetTransaction() = default;

  ~PipelineBudgetTransaction() { rollback(); }

  PipelineBudgetTransaction(const PipelineBudgetTransaction &) = delete;
  PipelineBudgetTransaction &
  operator=(const PipelineBudgetTransaction &) = delete;

  [[nodiscard]] bool begin(PreparedKernelTemplateRegistryState &state,
                           PreparedKernelTemplateRegistry &registry) noexcept {
    if (state_ != nullptr) {
      return false;
    }
    lock_ = std::unique_lock<std::recursive_mutex>{state.mutex};
    state_ = &state;
    registry_ = &registry;
    entry_count_ = state.entries.size();
    charge_count_ = state.template_charges.size();
    consumed_ = state.consumed;
    reservation_ = registry.reservation;
    return true;
  }

  void commit() noexcept {
    committed_ = true;
    if (lock_.owns_lock()) {
      lock_.unlock();
    }
  }

private:
  void rollback() noexcept {
    if (state_ == nullptr || committed_) {
      return;
    }
    // All registry publication is serialized by the recursively held lock, so
    // these tails belong exclusively to this failed cold preparation.
    state_->entries.resize(entry_count_);
    state_->template_charges.resize(charge_count_);
    state_->consumed = consumed_;
    registry_->reservation = reservation_;
    committed_ = true;
    if (lock_.owns_lock()) {
      lock_.unlock();
    }
  }

  PreparedKernelTemplateRegistryState *state_{};
  PreparedKernelTemplateRegistry *registry_{};
  std::unique_lock<std::recursive_mutex> lock_{};
  std::size_t entry_count_{};
  std::size_t charge_count_{};
  PreparedKernelPipelineReservation consumed_{};
  PreparedKernelPipelineReservation reservation_{};
  bool committed_{};
};

[[nodiscard]] PreparedKernelTemplateRegistryState *
registry_state(const PreparedKernelTemplateRegistry &registry) noexcept {
  auto *const state =
      static_cast<PreparedKernelTemplateRegistryState *>(registry.owner.get());
  return state != nullptr && state->magic == kTemplateRegistryMagic ? state
                                                                    : nullptr;
}

[[nodiscard]] bool add(const std::uint64_t left, const std::uint64_t right,
                       std::uint64_t &out) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  out = left + right;
  return true;
}

[[nodiscard]] bool accumulate(std::uint64_t &target,
                              const std::uint64_t value) noexcept {
  return add(target, value, target);
}

[[nodiscard]] bool multiply(const std::uint64_t left, const std::uint64_t right,
                            std::uint64_t &out) noexcept {
  if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  out = left * right;
  return true;
}

[[nodiscard]] bool accumulate_reservation(
    PreparedKernelPipelineReservation &target,
    const PreparedKernelPipelineReservation &value) noexcept {
  target.source_transient_bytes =
      std::max(target.source_transient_bytes, value.source_transient_bytes);
  target.host_transient_bytes =
      std::max(target.host_transient_bytes, value.host_transient_bytes);
  target.backend_command_binding_slot_upper =
      std::max(target.backend_command_binding_slot_upper,
               value.backend_command_binding_slot_upper);
  return accumulate(target.host_bytes, value.host_bytes) &&
         accumulate(target.native_bytes, value.native_bytes) &&
         accumulate(target.route_host_bytes, value.route_host_bytes) &&
         accumulate(target.route_native_bytes, value.route_native_bytes) &&
         accumulate(target.template_host_bytes, value.template_host_bytes) &&
         accumulate(target.template_native_bytes,
                    value.template_native_bytes) &&
         accumulate(target.template_source_bytes,
                    value.template_source_bytes) &&
         accumulate(target.route_count, value.route_count) &&
         accumulate(target.template_count, value.template_count) &&
         accumulate(target.template_step_count, value.template_step_count) &&
         accumulate(target.route_step_count, value.route_step_count) &&
         accumulate(target.authored_entry_count, value.authored_entry_count) &&
         accumulate(target.occurrence_count, value.occurrence_count) &&
         accumulate(target.window_count, value.window_count) &&
         accumulate(target.nested_group_count, value.nested_group_count) &&
         accumulate(target.backend_dispatch_count,
                    value.backend_dispatch_count) &&
         accumulate(target.backend_reset_dispatch_count,
                    value.backend_reset_dispatch_count) &&
         accumulate(target.backend_window_dispatch_count,
                    value.backend_window_dispatch_count) &&
         accumulate(target.backend_indirect_dispatch_count,
                    value.backend_indirect_dispatch_count) &&
         accumulate(target.backend_window_state_count,
                    value.backend_window_state_count) &&
         accumulate(target.backend_window_descriptor_state_count,
                    value.backend_window_descriptor_state_count) &&
         accumulate(target.backend_step_occurrence_count,
                    value.backend_step_occurrence_count) &&
         accumulate(target.backend_step_description_count,
                    value.backend_step_description_count) &&
         accumulate(target.backend_status_source_count,
                    value.backend_status_source_count) &&
         accumulate(target.backend_status_entry_count,
                    value.backend_status_entry_count) &&
         accumulate(target.backend_telemetry_count,
                    value.backend_telemetry_count) &&
         accumulate(target.backend_status_command_count,
                    value.backend_status_command_count) &&
         accumulate(target.backend_telemetry_command_count,
                    value.backend_telemetry_command_count) &&
         accumulate(target.backend_publication_count,
                    value.backend_publication_count) &&
         accumulate(target.backend_terminal_publication_count,
                    value.backend_terminal_publication_count) &&
         accumulate(target.backend_window_control_command_count,
                    value.backend_window_control_command_count) &&
         accumulate(target.backend_publication_command_count,
                    value.backend_publication_command_count) &&
         accumulate(target.backend_command_count,
                    value.backend_command_count) &&
         accumulate(target.backend_command_chunk_count,
                    value.backend_command_chunk_count) &&
         accumulate(target.backend_command_native_bytes,
                    value.backend_command_native_bytes) &&
         accumulate(target.backend_command_binding_count,
                    value.backend_command_binding_count) &&
         accumulate(target.backend_parameter_bytes,
                    value.backend_parameter_bytes) &&
         accumulate(target.backend_profile_step_count,
                    value.backend_profile_step_count) &&
         accumulate(target.backend_profile_command_count,
                    value.backend_profile_command_count) &&
         accumulate(target.backend_query_count, value.backend_query_count) &&
         accumulate(target.backend_native_buffer_count,
                    value.backend_native_buffer_count) &&
         accumulate(target.backend_native_object_count,
                    value.backend_native_object_count) &&
         accumulate(target.descriptor_set_count, value.descriptor_set_count) &&
         accumulate(target.descriptor_count, value.descriptor_count) &&
         accumulate(target.native_allocation_count,
                    value.native_allocation_count) &&
         accumulate(target.map_recurrence.route_host_bytes,
                    value.map_recurrence.route_host_bytes) &&
         accumulate(target.map_recurrence.route_native_bytes,
                    value.map_recurrence.route_native_bytes) &&
         accumulate(target.map_recurrence.template_host_bytes,
                    value.map_recurrence.template_host_bytes) &&
         accumulate(target.map_recurrence.template_native_bytes,
                    value.map_recurrence.template_native_bytes) &&
         accumulate(target.map_recurrence.template_source_bytes,
                    value.map_recurrence.template_source_bytes) &&
         accumulate(target.map_recurrence.group_count,
                    value.map_recurrence.group_count) &&
         accumulate(target.map_recurrence.history_group_count,
                    value.map_recurrence.history_group_count) &&
         accumulate(target.map_recurrence.template_count,
                    value.map_recurrence.template_count) &&
         accumulate(target.map_recurrence.terminal_template_group_capacity,
                    value.map_recurrence.terminal_template_group_capacity) &&
         accumulate(target.map_recurrence.history_template_group_capacity,
                    value.map_recurrence.history_template_group_capacity) &&
         accumulate(target.map_recurrence.route_step_count,
                    value.map_recurrence.route_step_count) &&
         accumulate(target.map_recurrence.template_step_count,
                    value.map_recurrence.template_step_count) &&
         accumulate(target.map_recurrence.descriptor_set_count,
                    value.map_recurrence.descriptor_set_count) &&
         accumulate(target.map_recurrence.descriptor_count,
                    value.map_recurrence.descriptor_count) &&
         accumulate(target.map_recurrence.route_native_allocation_count,
                    value.map_recurrence.route_native_allocation_count) &&
         accumulate(target.map_recurrence.template_native_allocation_count,
                    value.map_recurrence.template_native_allocation_count) &&
         (target.map_recurrence.source_transient_bytes =
              std::max(target.map_recurrence.source_transient_bytes,
                       value.map_recurrence.source_transient_bytes),
          true);
}

[[nodiscard]] bool valid_map_recurrence_reservation(
    const MapRecurrencePreparationPlan &plan,
    const PreparedMapRecurrenceReservation &reservation) noexcept {
  const std::uint64_t expected_templates =
      static_cast<std::uint64_t>(plan.terminal_group_count() != 0u) +
      static_cast<std::uint64_t>(plan.history_group_count != 0u);
  return plan.eligible() && reservation.group_count == plan.group_count &&
         reservation.history_group_count == plan.history_group_count &&
         reservation.template_count == expected_templates &&
         reservation.terminal_template_group_capacity ==
             plan.terminal_template_group_capacity &&
         reservation.history_template_group_capacity ==
             plan.history_template_group_capacity &&
         reservation.route_step_count == plan.group_count &&
         reservation.template_step_count == expected_templates;
}

[[nodiscard]] rund::AccelCheck plan_map_recurrence_reservation(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
  if (!plan.ok) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  if (!plan.eligible()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (ops.plan_pipeline_recurrence == nullptr) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const rund::AccelCheck checked =
      ops.plan_pipeline_recurrence(plan, reservation);
  if (!checked.ok || !valid_map_recurrence_reservation(plan, reservation)) {
    reservation = {};
    return rund::AccelCheck{false, checked.reason == nullptr
                                       ? "compute_pipeline_capacity"
                                       : checked.reason};
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] bool scale_map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value, const std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept {
  if (copies == 0u) {
    return false;
  }
  scaled = {};
  return multiply(value.route_host_bytes, copies, scaled.route_host_bytes) &&
         multiply(value.route_native_bytes, copies,
                  scaled.route_native_bytes) &&
         multiply(value.group_count, copies, scaled.group_count) &&
         multiply(value.history_group_count, copies,
                  scaled.history_group_count) &&
         multiply(value.route_step_count, copies, scaled.route_step_count) &&
         multiply(value.route_native_allocation_count, copies,
                  scaled.route_native_allocation_count);
}

[[nodiscard]] PreparedMapRecurrenceReservation map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept {
  return PreparedMapRecurrenceReservation{
      .route_host_bytes = value.route_host_bytes,
      .route_native_bytes = value.route_native_bytes,
      .group_count = value.group_count,
      .history_group_count = value.history_group_count,
      .route_step_count = value.route_step_count,
      .route_native_allocation_count = value.route_native_allocation_count,
  };
}

[[nodiscard]] PreparedMapRecurrenceReservation
map_recurrence_template_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept {
  return PreparedMapRecurrenceReservation{
      .template_host_bytes = value.template_host_bytes,
      .template_native_bytes = value.template_native_bytes,
      .template_source_bytes = value.template_source_bytes,
      .source_transient_bytes = value.source_transient_bytes,
      .template_count = value.template_count,
      .terminal_template_group_capacity =
          value.terminal_template_group_capacity,
      .history_template_group_capacity = value.history_template_group_capacity,
      .template_step_count = value.template_step_count,
      .descriptor_set_count = value.descriptor_set_count,
      .descriptor_count = value.descriptor_count,
      .template_native_allocation_count =
          value.template_native_allocation_count,
  };
}

[[nodiscard]] bool same_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &left,
    const PreparedMapRecurrenceReservation &right) noexcept {
  return left.route_host_bytes == right.route_host_bytes &&
         left.route_native_bytes == right.route_native_bytes &&
         left.template_host_bytes == right.template_host_bytes &&
         left.template_native_bytes == right.template_native_bytes &&
         left.template_source_bytes == right.template_source_bytes &&
         left.source_transient_bytes == right.source_transient_bytes &&
         left.group_count == right.group_count &&
         left.history_group_count == right.history_group_count &&
         left.template_count == right.template_count &&
         left.terminal_template_group_capacity ==
             right.terminal_template_group_capacity &&
         left.history_template_group_capacity ==
             right.history_template_group_capacity &&
         left.route_step_count == right.route_step_count &&
         left.template_step_count == right.template_step_count &&
         left.descriptor_set_count == right.descriptor_set_count &&
         left.descriptor_count == right.descriptor_count &&
         left.route_native_allocation_count ==
             right.route_native_allocation_count &&
         left.template_native_allocation_count ==
             right.template_native_allocation_count;
}

[[nodiscard]] bool accumulate_map_recurrence_template(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept {
  target.source_transient_bytes =
      std::max(target.source_transient_bytes, value.source_transient_bytes);
  return accumulate(target.template_host_bytes, value.template_host_bytes) &&
         accumulate(target.template_native_bytes,
                    value.template_native_bytes) &&
         accumulate(target.template_source_bytes,
                    value.template_source_bytes) &&
         accumulate(target.template_count, value.template_count) &&
         accumulate(target.terminal_template_group_capacity,
                    value.terminal_template_group_capacity) &&
         accumulate(target.history_template_group_capacity,
                    value.history_template_group_capacity) &&
         accumulate(target.template_step_count, value.template_step_count) &&
         accumulate(target.descriptor_set_count, value.descriptor_set_count) &&
         accumulate(target.descriptor_count, value.descriptor_count) &&
         accumulate(target.template_native_allocation_count,
                    value.template_native_allocation_count);
}

[[nodiscard]] bool accumulate_map_recurrence_route(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept {
  return accumulate(target.route_host_bytes, value.route_host_bytes) &&
         accumulate(target.route_native_bytes, value.route_native_bytes) &&
         accumulate(target.group_count, value.group_count) &&
         accumulate(target.history_group_count, value.history_group_count) &&
         accumulate(target.route_step_count, value.route_step_count) &&
         accumulate(target.route_native_allocation_count,
                    value.route_native_allocation_count);
}

[[nodiscard]] rund::AccelCheck plan_map_recurrence_template_variant(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    const bool history,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
  const bool present = history ? plan.history_group_count != 0u
                               : plan.terminal_group_count() != 0u;
  if (!plan.ok || !present) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  MapRecurrencePreparationPlan variant = plan;
  variant.group_count = 1u;
  variant.history_group_count = history ? 1u : 0u;
  if (history) {
    variant.terminal_source = {};
    variant.terminal_template_group_capacity = 0u;
  } else {
    variant.history_source = {};
    variant.history_template_group_capacity = 0u;
  }
  PreparedMapRecurrenceReservation planned{};
  const rund::AccelCheck checked =
      plan_map_recurrence_reservation(ops, variant, planned);
  if (!checked.ok) {
    return checked;
  }
  reservation = map_recurrence_template_reservation(planned);
  if (reservation.template_count != 1u ||
      reservation.template_step_count != 1u) {
    reservation = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck verify_map_recurrence_template_variants(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    const PreparedMapRecurrenceReservation &combined,
    PreparedMapRecurrenceReservation &terminal,
    PreparedMapRecurrenceReservation &history) noexcept {
  terminal = {};
  history = {};
  PreparedMapRecurrenceReservation observed{};
  if (plan.terminal_group_count() != 0u) {
    const rund::AccelCheck checked =
        plan_map_recurrence_template_variant(ops, plan, false, terminal);
    if (!checked.ok ||
        !accumulate_map_recurrence_template(observed, terminal)) {
      return checked.ok ? rund::AccelCheck{false, "compute_pipeline_capacity"}
                        : checked;
    }
  }
  if (plan.history_group_count != 0u) {
    const rund::AccelCheck checked =
        plan_map_recurrence_template_variant(ops, plan, true, history);
    if (!checked.ok || !accumulate_map_recurrence_template(observed, history)) {
      return checked.ok ? rund::AccelCheck{false, "compute_pipeline_capacity"}
                        : checked;
    }
  }
  return same_map_recurrence_reservation(
             observed, map_recurrence_template_reservation(combined))
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

// Projects the auditable recurrence subset into the generic preparation
// totals exactly once. Callers then use the ordinary reservation accumulator;
// no backend may maintain a second hidden recurrence budget.
[[nodiscard]] bool project_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &recurrence,
    PreparedKernelPipelineReservation &projection) noexcept {
  projection = {};
  projection.ok = true;
  projection.reason = "ok";
  projection.map_recurrence = recurrence;
  projection.source_transient_bytes = recurrence.source_transient_bytes;
  return add(recurrence.route_host_bytes, recurrence.template_host_bytes,
             projection.host_bytes) &&
         add(recurrence.route_native_bytes, recurrence.template_native_bytes,
             projection.native_bytes) &&
         accumulate(projection.route_host_bytes, recurrence.route_host_bytes) &&
         accumulate(projection.route_native_bytes,
                    recurrence.route_native_bytes) &&
         accumulate(projection.template_host_bytes,
                    recurrence.template_host_bytes) &&
         accumulate(projection.template_native_bytes,
                    recurrence.template_native_bytes) &&
         accumulate(projection.template_source_bytes,
                    recurrence.template_source_bytes) &&
         accumulate(projection.route_count, recurrence.group_count) &&
         accumulate(projection.template_count, recurrence.template_count) &&
         accumulate(projection.route_step_count, recurrence.route_step_count) &&
         accumulate(projection.template_step_count,
                    recurrence.template_step_count) &&
         accumulate(projection.descriptor_set_count,
                    recurrence.descriptor_set_count) &&
         accumulate(projection.descriptor_count, recurrence.descriptor_count) &&
         accumulate(projection.native_allocation_count,
                    recurrence.route_native_allocation_count) &&
         accumulate(projection.native_allocation_count,
                    recurrence.template_native_allocation_count);
}

void fingerprint_mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
}

void fingerprint_pipeline_header(std::uint64_t &hi, std::uint64_t &lo,
                                 const rund::AccelContext &context,
                                 const PreparedKernelPipelineShape shape,
                                 const std::uint64_t route_count) noexcept {
  hi = 0x72756e442e6c696dull;
  lo = 0x69742e70726f6772ull;
  fingerprint_mix(hi, context.id);
  fingerprint_mix(lo, static_cast<std::uint64_t>(context.api));
  fingerprint_mix(hi, shape.publication_count);
  fingerprint_mix(lo, shape.terminal_publication_count);
  fingerprint_mix(hi, shape.backend_publication_command_count);
  fingerprint_mix(hi, shape.window_state_count);
  fingerprint_mix(lo, shape.window_descriptor_state_count);
  fingerprint_mix(hi, shape.publication_fingerprint_hi);
  fingerprint_mix(lo, shape.publication_fingerprint_lo);
  fingerprint_mix(hi, shape.declared_step_count);
  fingerprint_mix(hi, shape.route_copies);
  fingerprint_mix(hi, static_cast<std::uint64_t>(shape.profile_steps));
  fingerprint_mix(lo, route_count);
}

[[nodiscard]] PreparedKernelPublicationViewIdentity
publication_view_identity(const rund::kernel::ResidentBufferRef &view,
                          const std::uint32_t ordinal) noexcept {
  return PreparedKernelPublicationViewIdentity{
      .backing_bytes = view.bytes,
      .offset_bytes = view.offset_bytes,
      .count = view.count,
      .stride_bytes = view.stride_bytes,
      .element_bytes = view.element_bytes,
      .resource_ordinal = ordinal,
      .usage = view.usage,
  };
}

[[nodiscard]] PreparedKernelPipelineShape
runtime_pipeline_shape(const std::span<const BackendPublish> publications,
                       const std::span<const BackendRecurrence> recurrences,
                       const std::uint32_t declared_step_count,
                       const std::uint32_t route_copies,
                       const bool profile_steps) noexcept {
  PreparedKernelPipelineShape shape{
      .publication_count = publications.size(),
      .declared_step_count = declared_step_count,
      .route_copies = route_copies,
      .profile_steps = profile_steps,
  };
  SeedPreparedKernelPublicationFingerprint(shape.publication_fingerprint_hi,
                                           shape.publication_fingerprint_lo);
  for (const BackendPublish &publication : publications) {
    const bool window = publication.kind == BackendPublishKind::Window;
    std::uint64_t publication_commands = 0u;
    if (!PreparedKernelPublicationCommandContribution(
            window, publication.maximum, publication.tile,
            publication_commands) ||
        !rund::kernel::checked::add(shape.backend_publication_command_count,
                                    publication_commands,
                                    shape.backend_publication_command_count)) {
      shape.backend_publication_command_count =
          std::numeric_limits<std::uint64_t>::max();
    }
    shape.terminal_publication_count += window ? 0u : 1u;
    PreparedKernelPublicationIdentity identity{
        .count = publication_view_identity(publication.count.source,
                                           publication.count_ordinal),
        .target = publication_view_identity(publication.target,
                                            publication.target_ordinal),
        .state = publication.state,
        .final = publication.final,
        .maximum = publication.maximum,
        .tile = publication.tile,
        .kind = static_cast<std::uint8_t>(publication.kind),
    };
    for (std::size_t bank = 0u; bank < publication.sources.size(); ++bank) {
      identity.sources[bank] = publication_view_identity(
          publication.sources[bank].source, publication.source_ordinals[bank]);
    }
    MixPreparedKernelPublicationFingerprint(shape.publication_fingerprint_hi,
                                            shape.publication_fingerprint_lo,
                                            identity);
  }
  for (std::size_t index = 0u; index < recurrences.size(); ++index) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr) {
      continue;
    }
    shape.window_state_count =
        std::max(shape.window_state_count,
                 static_cast<std::uint64_t>(window->state) + 1u);
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const BackendWindow *const previous = recurrences[prior].window;
      if (previous != nullptr && previous->state == window->state) {
        first = false;
        break;
      }
    }
    shape.window_descriptor_state_count += first ? 1u : 0u;
  }
  return shape;
}

[[nodiscard]] bool
valid_publication_shape(const PreparedKernelPipelineShape &shape) noexcept {
  std::uint64_t minimum_commands = 0u;
  return shape.terminal_publication_count <= shape.publication_count &&
         rund::kernel::checked::add(shape.publication_count,
                                    shape.terminal_publication_count,
                                    minimum_commands) &&
         shape.backend_publication_command_count >= minimum_commands &&
         ((shape.publication_count == 0u) ==
          (shape.backend_publication_command_count == 0u));
}

void fingerprint_layout(std::uint64_t &hi, std::uint64_t &lo,
                        const KernelViewLayout *const views,
                        const KernelScratchLayout *const scratch) noexcept {
  fingerprint_mix(hi, views == nullptr
                          ? std::numeric_limits<std::uint64_t>::max()
                          : views->size());
  if (views != nullptr) {
    for (const KernelViewSlot &view : *views) {
      fingerprint_mix(lo, view.binding);
      fingerprint_mix(hi, view.slot);
      fingerprint_mix(lo, view.backing_bytes);
      fingerprint_mix(hi, view.offset_bytes);
      fingerprint_mix(lo, view.count);
      fingerprint_mix(hi, view.stride_bytes);
      fingerprint_mix(lo, view.element_bytes);
      fingerprint_mix(hi, view.usage);
    }
  }
  fingerprint_mix(lo, scratch == nullptr
                          ? std::numeric_limits<std::uint64_t>::max()
                          : scratch->size());
  if (scratch != nullptr) {
    for (const KernelScratchPage &page : *scratch) {
      fingerprint_mix(hi, page.slot);
      fingerprint_mix(lo, page.bytes);
    }
  }
}

void fingerprint_route(
    std::uint64_t &hi, std::uint64_t &lo, const std::uint64_t kernel_id,
    const std::uint64_t graph_id_hi, const std::uint64_t graph_id_lo,
    const std::uint64_t node_count, const rund::AccelApi api,
    const std::uint64_t tile_count, const KernelViewLayout *const views,
    const KernelScratchLayout *const scratch, const std::uint64_t entry_count,
    const std::uint64_t occurrence_count, const std::uint64_t window_count,
    const std::uint64_t nested_group_count,
    const std::uint64_t map_recurrence_group_count,
    const std::uint64_t map_recurrence_history_group_count,
    const std::uint64_t recurrence_hi, const std::uint64_t recurrence_lo,
    const std::uint32_t route_copies) noexcept {
  fingerprint_mix(hi, kernel_id);
  fingerprint_mix(lo, graph_id_hi);
  fingerprint_mix(hi, graph_id_lo);
  fingerprint_mix(lo, node_count);
  fingerprint_mix(hi, static_cast<std::uint64_t>(api));
  fingerprint_mix(lo, tile_count);
  fingerprint_layout(hi, lo, views, scratch);
  fingerprint_mix(lo, entry_count);
  fingerprint_mix(hi, occurrence_count);
  fingerprint_mix(lo, window_count);
  fingerprint_mix(hi, nested_group_count);
  fingerprint_mix(lo, map_recurrence_group_count);
  fingerprint_mix(hi, map_recurrence_history_group_count);
  fingerprint_mix(lo, recurrence_hi);
  fingerprint_mix(hi, recurrence_lo);
  fingerprint_mix(lo, route_copies);
}

struct ExpandedPipeline final {
  std::vector<BackendBatchEntry> commands;
  std::vector<std::uint8_t> barriers;
  std::vector<BackendWindow> windows;
  std::vector<TileTransducer> transducers;
  std::vector<NestedAggregate> aggregates;
  std::uint32_t command_count{};
  bool compact_aggregate{};
  const char *reason = "accel_kernel_run_invalid";
  PreparedPipelineFailureContext failure{};
};

class BackendPreparationCursor final {
public:
  BackendPreparationCursor(BackendRun &run,
                           PreparedKernelTemplateRegistry &templates,
                           std::uint32_t &failed_node,
                           const BackendTemplateRouteDemand demand) noexcept
      : run_{run} {
    run_.templates = &templates;
    run_.template_route_demand = demand;
    run_.failed_node = &failed_node;
  }

  ~BackendPreparationCursor() {
    run_.templates = nullptr;
    run_.template_route_demand = {};
    run_.failed_node = nullptr;
  }

  BackendPreparationCursor(const BackendPreparationCursor &) = delete;
  BackendPreparationCursor &
  operator=(const BackendPreparationCursor &) = delete;

private:
  BackendRun &run_;
};

[[nodiscard]] bool
backend_template_route_demand(const std::uint64_t owner_count,
                              const std::uint64_t route_copies,
                              BackendTemplateRouteDemand &demand) noexcept {
  std::uint64_t capacity = 0u;
  if (owner_count == 0u ||
      owner_count > std::numeric_limits<std::uint32_t>::max() ||
      (route_copies != 1u && route_copies != 2u) ||
      !rund::kernel::checked::mul(owner_count, route_copies, capacity) ||
      capacity > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const BackendTemplateRouteDemand candidate{
      .owner_count = static_cast<std::uint32_t>(owner_count),
      .route_copies = static_cast<std::uint32_t>(route_copies),
      .capacity = static_cast<std::uint32_t>(capacity),
  };
  if (!candidate.valid()) {
    return false;
  }
  demand = candidate;
  return true;
}

// Freezes one exact demand for every route entry before the first native
// materialization. Duplicate entries that borrow one RunState do not increase
// demand. Structurally equal but independently bound RunStates do. Equality is
// checked in both directions and must form one complete equivalence class;
// an asymmetric or non-transitive backend predicate fails closed.
[[nodiscard]] bool plan_backend_template_route_demands(
    const std::span<const PreparedKernelRun *const> runs,
    const std::uint32_t route_copies,
    const std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept {
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      demands.size() != runs.size() ||
      (route_copies != 1u && route_copies != 2u)) {
    return false;
  }

  std::array<std::size_t, PreparedPipelineStepCapacity> unique_entries{};
  std::array<std::size_t, PreparedPipelineStepCapacity> entry_unique{};
  std::array<std::size_t, PreparedPipelineStepCapacity> unique_group{};
  std::array<std::uint64_t, PreparedPipelineStepCapacity> group_counts{};
  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      candidates{};
  const BackendOps *common_ops = nullptr;
  std::size_t unique_count = 0u;

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const ops =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr || ops == nullptr ||
        ops->same_pipeline_template == nullptr ||
        !IsPipelinePrivatePreparation(state->mode) ||
        (common_ops != nullptr && common_ops != ops)) {
      return false;
    }
    common_ops = ops;

    std::size_t duplicate = unique_count;
    for (std::size_t prior = 0u; prior < unique_count; ++prior) {
      const PreparedKernelRun *const previous = runs[unique_entries[prior]];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate = prior;
        break;
      }
    }
    if (duplicate != unique_count) {
      entry_unique[index] = duplicate;
      continue;
    }

    std::size_t matched_group = unique_count;
    std::uint64_t matched_members = 0u;
    const BackendRun &run = state->bound.run;
    if (!ops->same_pipeline_template(run, run)) {
      return false;
    }
    for (std::size_t prior = 0u; prior < unique_count; ++prior) {
      const PreparedKernelRun *const previous = runs[unique_entries[prior]];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      if (previous_state == nullptr) {
        return false;
      }
      const BackendRun &prior_run = previous_state->bound.run;
      const bool forward = ops->same_pipeline_template(run, prior_run);
      const bool reverse = ops->same_pipeline_template(prior_run, run);
      if (forward != reverse) {
        return false;
      }
      if (!forward) {
        continue;
      }
      ++matched_members;
      const std::size_t prior_group = unique_group[prior];
      if (matched_group == unique_count) {
        matched_group = prior_group;
      } else if (matched_group != prior_group) {
        return false;
      }
    }
    if (matched_group == unique_count) {
      matched_group = unique_count;
    } else if (matched_members != group_counts[matched_group]) {
      return false;
    }
    unique_entries[unique_count] = index;
    entry_unique[index] = unique_count;
    unique_group[unique_count] = matched_group;
    ++group_counts[matched_group];
    ++unique_count;
  }

  std::uint64_t groups = 0u;
  for (std::size_t unique = 0u; unique < unique_count; ++unique) {
    const std::size_t group = unique_group[unique];
    if (group >= unique_count || group_counts[group] == 0u ||
        !backend_template_route_demand(group_counts[group], route_copies,
                                       candidates[unique_entries[unique]])) {
      return false;
    }
    groups += static_cast<std::uint64_t>(group == unique);
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const std::size_t unique = entry_unique[index];
    if (unique >= unique_count) {
      return false;
    }
    candidates[index] = candidates[unique_entries[unique]];
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    demands[index] = candidates[index];
  }
  unique_route_count = unique_count;
  template_count = groups;
  return true;
}

[[nodiscard]] bool same_nested_window(const BackendWindow &left,
                                      const BackendWindow &right) noexcept {
  return left.state == right.state && left.maximum == right.maximum &&
         left.tile == right.tile && left.expected == right.expected &&
         left.outer_bound == right.outer_bound &&
         left.inner_bound == right.inner_bound &&
         left.has_terminal == right.has_terminal &&
         left.count.source.id == right.count.source.id &&
         left.count.source.offset_bytes == right.count.source.offset_bytes &&
         left.count.handle == right.count.handle;
}

[[nodiscard]] bool
nested_shape(const std::span<const BackendBatchEntry> templates,
             const std::size_t first, std::size_t &end,
             std::uint32_t &outer_bound, std::uint32_t &inner_bound) noexcept {
  if (first >= templates.size()) {
    return false;
  }
  const BackendWindow *const first_window = templates[first].recurrence.window;
  if (first_window == nullptr ||
      first_window->phase != BackendWindowPhase::NestedSeed ||
      first_window->outer_bound == 0u) {
    return false;
  }
  outer_bound = first_window->outer_bound;
  inner_bound = first_window->inner_bound;
  const std::uint64_t group_count =
      static_cast<std::uint64_t>(outer_bound) + inner_bound + 3u;
  if (group_count > templates.size() - first) {
    return false;
  }
  end = first + static_cast<std::size_t>(group_count);
  for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
    const BackendWindow *const window =
        templates[first + outer].recurrence.window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedSeed ||
        window->outer_iteration != outer || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t action_first = first + outer_bound;
  for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
    const BackendWindow *const window =
        templates[action_first + inner].recurrence.window;
    if (window == nullptr ||
        window->phase != BackendWindowPhase::NestedAction ||
        window->inner_iteration != inner || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t fold_first = action_first + inner_bound;
  for (std::uint32_t route = 0u; route < 3u; ++route) {
    const BackendWindow *const window =
        templates[fold_first + route].recurrence.window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedFold ||
        window->route != route || !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
nested_shape(const std::span<const BackendRecurrence> recurrences,
             const std::size_t first, std::size_t &end,
             std::uint32_t &outer_bound, std::uint32_t &inner_bound) noexcept {
  if (first >= recurrences.size()) {
    return false;
  }
  const BackendWindow *const first_window = recurrences[first].window;
  if (first_window == nullptr ||
      first_window->phase != BackendWindowPhase::NestedSeed ||
      first_window->outer_bound == 0u) {
    return false;
  }
  outer_bound = first_window->outer_bound;
  inner_bound = first_window->inner_bound;
  const std::uint64_t group_count =
      static_cast<std::uint64_t>(outer_bound) + inner_bound + 3u;
  if (group_count > recurrences.size() - first) {
    return false;
  }
  end = first + static_cast<std::size_t>(group_count);
  for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
    const BackendWindow *const window = recurrences[first + outer].window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedSeed ||
        window->outer_iteration != outer || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t action_first = first + outer_bound;
  for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
    const BackendWindow *const window =
        recurrences[action_first + inner].window;
    if (window == nullptr ||
        window->phase != BackendWindowPhase::NestedAction ||
        window->inner_iteration != inner || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t fold_first = action_first + inner_bound;
  for (std::uint32_t route = 0u; route < 3u; ++route) {
    const BackendWindow *const window = recurrences[fold_first + route].window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedFold ||
        window->route != route || !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] PreparedKernelRecurrenceIdentity
recurrence_identity(const BackendRecurrence &recurrence) noexcept {
  PreparedKernelRecurrenceIdentity identity{
      .logical_step = recurrence.logical_step,
      .iteration = recurrence.iteration,
      .bound = recurrence.bound,
      .writes_each_iteration = recurrence.writes_each_iteration,
  };
  const BackendWindow *const window = recurrence.window;
  if (window == nullptr) {
    return identity;
  }
  identity.maximum = window->maximum;
  identity.tile = window->tile;
  identity.expected = window->expected;
  identity.outer_iteration = window->outer_iteration;
  identity.outer_bound = window->outer_bound;
  identity.inner_iteration = window->inner_iteration;
  identity.inner_bound = window->inner_bound;
  identity.route = window->route;
  identity.state = window->state;
  identity.phase = static_cast<std::uint8_t>(window->phase);
  identity.has_window = true;
  identity.has_terminal = window->has_terminal;
  return identity;
}

[[nodiscard]] bool recurrence_occurrences(const BackendRecurrence &recurrence,
                                          std::uint64_t &count) noexcept {
  const BackendWindow *const window = recurrence.window;
  if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
    count = 1u;
    return true;
  }
  switch (window->phase) {
  case BackendWindowPhase::NestedSeed:
    count = 1u;
    return true;
  case BackendWindowPhase::NestedAction:
    count = window->outer_bound;
    return count != 0u;
  case BackendWindowPhase::NestedFold:
    if (window->route == 0u) {
      count = 1u;
      return true;
    }
    if (window->route == 1u) {
      count = window->outer_bound / 2u;
      return true;
    }
    if (window->route == 2u && window->outer_bound != 0u) {
      count = (window->outer_bound - 1u) / 2u;
      return true;
    }
    return false;
  case BackendWindowPhase::Ordinary:
    break;
  }
  return false;
}

[[nodiscard]] bool route_recurrence_shape(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const void *const route_owner, std::uint64_t &entry_count,
    std::uint64_t &occurrence_count, std::uint64_t &window_count,
    std::uint64_t &nested_group_count,
    std::uint64_t &map_recurrence_group_count,
    std::uint64_t &map_recurrence_history_group_count,
    std::uint64_t &recurrence_hi, std::uint64_t &recurrence_lo) noexcept {
  if (route_owner == nullptr || runs.size() != recurrences.size()) {
    return false;
  }
  SeedPreparedKernelRecurrenceFingerprint(recurrence_hi, recurrence_lo);
  bool top_level_recurrence =
      recurrences.size() > 1u &&
      recurrences.size() <= std::numeric_limits<std::uint32_t>::max();
  const std::uint32_t top_logical =
      top_level_recurrence ? recurrences.front().logical_step : 0u;
  const bool top_history =
      top_level_recurrence && recurrences.front().writes_each_iteration;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const BackendRecurrence &marker = recurrences[index];
    top_level_recurrence =
        top_level_recurrence && marker.logical_step == top_logical &&
        marker.iteration == index && marker.bound == recurrences.size() &&
        marker.window == nullptr && marker.writes_each_iteration == top_history;
    const PreparedKernelRun *const item = runs[index];
    if (item == nullptr || item->owner.get() != route_owner) {
      continue;
    }
    std::uint64_t occurrences = 0u;
    if (!recurrence_occurrences(recurrences[index], occurrences) ||
        !accumulate(entry_count, 1u) ||
        !accumulate(occurrence_count, occurrences) ||
        (recurrences[index].window != nullptr &&
         !accumulate(window_count, occurrences))) {
      return false;
    }
    MixPreparedKernelRecurrenceFingerprint(
        recurrence_hi, recurrence_lo, recurrence_identity(recurrences[index]));
  }
  if (top_level_recurrence && runs.front() != nullptr &&
      runs.front()->owner.get() == route_owner &&
      (!accumulate(map_recurrence_group_count, 1u) ||
       (top_history && !accumulate(map_recurrence_history_group_count, 1u)))) {
    return false;
  }
  for (std::size_t index = 0u; index < recurrences.size();) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      ++index;
      continue;
    }
    std::size_t end = 0u;
    std::uint32_t outer = 0u;
    std::uint32_t inner = 0u;
    if (window->phase != BackendWindowPhase::NestedSeed ||
        !nested_shape(recurrences, index, end, outer, inner) ||
        runs[index] == nullptr) {
      return false;
    }
    if (runs[index]->owner.get() == route_owner &&
        !accumulate(nested_group_count, 1u)) {
      return false;
    }
    const std::size_t action_first = index + outer;
    if (inner > 1u && action_first < runs.size() &&
        runs[action_first] != nullptr &&
        runs[action_first]->owner.get() == route_owner &&
        !accumulate(map_recurrence_group_count, 1u)) {
      return false;
    }
    index = end;
  }
  // A compact terminal bank can be semantically authored yet have zero
  // physical occurrences for a one-iteration bound (for example fold bank
  // two). Its route/template identity still participates in the frozen plan.
  return entry_count != 0u;
}

struct MapRecurrenceTemplateCapacities final {
  std::uint64_t terminal{};
  std::uint64_t history{};
};

struct RuntimeRecurrenceRoutePlan final {
  std::uint64_t entry_count{};
  std::uint64_t occurrence_count{};
  std::uint64_t window_count{};
  std::uint64_t nested_group_count{};
  std::uint64_t group_count{};
  std::uint64_t history_group_count{};
  std::uint64_t fingerprint_hi{};
  std::uint64_t fingerprint_lo{};
  MapRecurrenceTemplateCapacities template_capacities{};
  bool first_owner{};
};

// Recurrence route state belongs to one prepared stream, while a terminal or
// history template is shared by every structurally equal route in both native
// generation streams. Freeze that complete equivalence-class demand before a
// backend planner can allocate a descriptor arena or another template-private
// owner. The existing route-demand planner remains the single authority for
// duplicate owners and symmetric/transitive template equality.
[[nodiscard]] bool plan_runtime_recurrence_routes(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::uint32_t route_copies,
    const std::span<RuntimeRecurrenceRoutePlan> plans) noexcept {
  if (runs.empty() || runs.size() != recurrences.size() ||
      plans.size() != runs.size() ||
      (route_copies != 1u && route_copies != 2u)) {
    return false;
  }

  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      demands{};
  std::uint64_t unique_route_count = 0u;
  std::uint64_t template_count = 0u;
  if (!plan_backend_template_route_demands(
          runs, route_copies,
          std::span<BackendTemplateRouteDemand>{demands.data(), runs.size()},
          unique_route_count, template_count) ||
      unique_route_count == 0u || template_count == 0u) {
    return false;
  }

  for (RuntimeRecurrenceRoutePlan &plan : plans) {
    plan = {};
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    if (item == nullptr || item->owner == nullptr) {
      return false;
    }
    bool duplicate = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    RuntimeRecurrenceRoutePlan &plan = plans[index];
    plan.first_owner = true;
    if (!route_recurrence_shape(runs, recurrences, item->owner.get(),
                                plan.entry_count, plan.occurrence_count,
                                plan.window_count, plan.nested_group_count,
                                plan.group_count, plan.history_group_count,
                                plan.fingerprint_hi, plan.fingerprint_lo) ||
        plan.history_group_count > plan.group_count) {
      return false;
    }
  }

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    RuntimeRecurrenceRoutePlan &plan = plans[index];
    if (!plan.first_owner) {
      continue;
    }
    const auto *const state =
        static_cast<const prepared::RunState *>(runs[index]->owner.get());
    if (state == nullptr) {
      return false;
    }
    const MapRecurrencePreparationPlan route_plan =
        PlanMapRecurrencePreparation(state->bound.run, plan.group_count,
                                     plan.history_group_count);
    if (!route_plan.ok) {
      return false;
    }
    for (std::size_t candidate = 0u; candidate < runs.size(); ++candidate) {
      const RuntimeRecurrenceRoutePlan &other = plans[candidate];
      if (!other.first_owner) {
        continue;
      }
      const auto *const other_state =
          static_cast<const prepared::RunState *>(runs[candidate]->owner.get());
      if (other_state == nullptr ||
          other_state->bound.run.ops != state->bound.run.ops) {
        return false;
      }
      const MapRecurrencePreparationPlan candidate_plan =
          PlanMapRecurrencePreparation(other_state->bound.run,
                                       other.group_count,
                                       other.history_group_count);
      if (!candidate_plan.ok) {
        return false;
      }
      if (SameMapRecurrenceTemplate(route_plan, candidate_plan, false) &&
          !accumulate(plan.template_capacities.terminal,
                      candidate_plan.terminal_group_count())) {
        return false;
      }
      if (SameMapRecurrenceTemplate(route_plan, candidate_plan, true) &&
          !accumulate(plan.template_capacities.history,
                      candidate_plan.history_group_count)) {
        return false;
      }
    }
    if (!multiply(plan.template_capacities.terminal, route_copies,
                  plan.template_capacities.terminal) ||
        !multiply(plan.template_capacities.history, route_copies,
                  plan.template_capacities.history)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool same_program_recurrence_authority(
    const PreparedKernelProgramRoute &left_route,
    const PreparedKernelProgramRoute &right_route) noexcept {
  const rund::AccelKernel *const left = left_route.kernel;
  const rund::AccelKernel *const right = right_route.kernel;
  return left != nullptr && right != nullptr && left->owner != nullptr &&
         left->owner == right->owner && left->kernel_id == right->kernel_id &&
         left->context_id == right->context_id &&
         left->graph_id_hi == right->graph_id_hi &&
         left->graph_id_lo == right->graph_id_lo &&
         left->node_count == right->node_count && left->api == right->api &&
         left->scalar == right->scalar && left->domain == right->domain;
}

[[nodiscard]] bool program_recurrence_template_capacities(
    const KernelExecution &execution,
    const std::span<const PreparedKernelProgramRoute> routes,
    const std::size_t route_index,
    const MapRecurrencePreparationPlan &route_plan,
    MapRecurrenceTemplateCapacities &capacities) noexcept {
  capacities = {};
  if (route_index >= routes.size() || !route_plan.ok) {
    return false;
  }
  const PreparedKernelProgramRoute &route = routes[route_index];
  for (const PreparedKernelProgramRoute &candidate : routes) {
    if (candidate.map_recurrence_history_group_count >
            candidate.map_recurrence_group_count ||
        (candidate.route_copies != 1u && candidate.route_copies != 2u)) {
      return false;
    }
    if (!same_program_recurrence_authority(route, candidate)) {
      continue;
    }
    const MapRecurrencePreparationPlan candidate_plan =
        PlanMapRecurrencePreparation(execution, candidate);
    if (!candidate_plan.ok) {
      return false;
    }
    std::uint64_t terminal = 0u;
    std::uint64_t history = 0u;
    if (SameMapRecurrenceTemplate(route_plan, candidate_plan, false) &&
        (!multiply(candidate_plan.terminal_group_count(),
                   candidate.route_copies, terminal) ||
         !accumulate(capacities.terminal, terminal))) {
      return false;
    }
    if (SameMapRecurrenceTemplate(route_plan, candidate_plan, true) &&
        (!multiply(candidate_plan.history_group_count, candidate.route_copies,
                   history) ||
         !accumulate(capacities.history, history))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] PreparedKernelPipelineReservation plan_pipeline_structure_counts(
    const std::uint64_t authored_entry_count,
    const std::uint64_t occurrence_count, const std::uint64_t window_count,
    const std::uint64_t nested_group_count) noexcept {
  PreparedKernelPipelineReservation result{};
  result.ok = true;
  result.reason = "ok";
  result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  result.authored_entry_count = authored_entry_count;
  result.occurrence_count = occurrence_count;
  result.window_count = window_count;
  result.nested_group_count = nested_group_count;
  std::uint64_t bytes = 0u;
  std::uint64_t item = 0u;
  if (!multiply(result.authored_entry_count, sizeof(BackendBatchEntry), item) ||
      !accumulate(bytes, item) ||
      !multiply(result.occurrence_count, sizeof(BackendBatchEntry), item) ||
      !accumulate(bytes, item) ||
      !multiply(result.occurrence_count, sizeof(std::uint8_t), item) ||
      !accumulate(bytes, item) ||
      !multiply(result.window_count, sizeof(BackendWindow), item) ||
      !accumulate(bytes, item) ||
      !multiply(result.nested_group_count, sizeof(TileTransducer), item) ||
      !accumulate(bytes, item) ||
      !multiply(result.nested_group_count, sizeof(NestedAggregate), item) ||
      !accumulate(bytes, item) ||
      (result.nested_group_count != 0u &&
       (!multiply(result.authored_entry_count, sizeof(std::uint32_t), item) ||
        !accumulate(bytes, item)))) {
    result.ok = false;
    result.reason = "compute_pipeline_capacity";
    return result;
  }
  result.host_bytes = bytes;
  return result;
}

[[nodiscard]] PreparedKernelPipelineReservation plan_pipeline_structure(
    const std::span<const BackendRecurrence> recurrences) noexcept {
  std::uint64_t occurrence_count = 0u;
  std::uint64_t window_count = 0u;
  std::uint64_t nested_group_count = 0u;
  for (std::size_t index = 0u; index < recurrences.size();) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      if (!accumulate(occurrence_count, 1u) ||
          (window != nullptr && !accumulate(window_count, 1u))) {
        return PreparedKernelPipelineReservation{
            .reason = "compute_pipeline_capacity"};
      }
      ++index;
      continue;
    }
    std::size_t end = 0u;
    std::uint32_t outer = 0u;
    std::uint32_t inner = 0u;
    if (!nested_shape(recurrences, index, end, outer, inner)) {
      return PreparedKernelPipelineReservation{.reason =
                                                   "accel_kernel_run_invalid"};
    }
    std::uint64_t commands_per_outer = 0u;
    std::uint64_t commands = 0u;
    if (!add(inner, 2u, commands_per_outer) ||
        !multiply(outer, commands_per_outer, commands) ||
        !accumulate(occurrence_count, commands) ||
        !accumulate(window_count, commands) ||
        !accumulate(nested_group_count, 1u)) {
      return PreparedKernelPipelineReservation{.reason =
                                                   "compute_pipeline_capacity"};
    }
    index = end;
  }
  return plan_pipeline_structure_counts(recurrences.size(), occurrence_count,
                                        window_count, nested_group_count);
}

// Projects canonical, unfused route occurrences into one backend stream.
// `dispatch_count` is the backend's physical body/view-command upper; reset and
// control families remain independent below. Nested transducers may materialize
// fewer physical commands, but this sum never substitutes that later
// optimization for the authored route ownership proved here.
[[nodiscard]] bool
accumulate_route_projection(PreparedKernelPipelineReservation &projection,
                            const PreparedKernelRouteReservation &route,
                            const std::uint64_t compact_entry_count,
                            const std::uint64_t occurrence_count,
                            const std::uint64_t window_count) noexcept {
  if (route.route_step_count == 0u || route.dispatch_count == 0u ||
      compact_entry_count == 0u) {
    return false;
  }
  PreparedKernelPipelineReservation candidate = projection;
  candidate.backend_command_binding_slot_upper =
      std::max(candidate.backend_command_binding_slot_upper,
               route.capture_binding_slot_upper);
  std::uint64_t contribution = 0u;
  std::uint64_t capture_dispatch_count = 0u;
  if (!accumulate(candidate.occurrence_count, occurrence_count) ||
      !accumulate(candidate.host_transient_bytes, route.host_transient_bytes) ||
      !multiply(route.dispatch_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_dispatch_count, contribution) ||
      !multiply(route.reset_dispatch_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_reset_dispatch_count, contribution) ||
      !multiply(route.route_step_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_step_occurrence_count, contribution) ||
      !multiply(route.route_step_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_step_description_count, contribution) ||
      !multiply(route.status_entry_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_status_entry_count, contribution) ||
      !multiply(route.status_source_count, compact_entry_count, contribution) ||
      !accumulate(candidate.backend_status_source_count, contribution) ||
      !multiply(route.telemetry_source_count, compact_entry_count,
                contribution) ||
      !accumulate(candidate.backend_telemetry_count, contribution) ||
      !multiply(route.status_command_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_status_command_count, contribution) ||
      !multiply(route.telemetry_source_count, occurrence_count, contribution) ||
      !accumulate(candidate.backend_telemetry_command_count, contribution) ||
      !multiply(route.status_parameter_bytes, occurrence_count, contribution) ||
      !accumulate(candidate.backend_parameter_bytes, contribution) ||
      !add(route.capture_direct_dispatch_count,
           route.capture_indirect_dispatch_count, capture_dispatch_count) ||
      !multiply(capture_dispatch_count, window_count, contribution) ||
      !accumulate(candidate.backend_window_dispatch_count, contribution) ||
      !multiply(route.capture_indirect_dispatch_count, window_count,
                contribution) ||
      !accumulate(candidate.backend_indirect_dispatch_count, contribution)) {
    return false;
  }
  projection = candidate;
  return true;
}

[[nodiscard]] rund::AccelCheck finalize_pipeline_backend_structure(
    const rund::AccelContext &context, const BackendOps &ops,
    const PreparedKernelPipelineReservation &projection,
    const std::uint64_t publication_count,
    const std::uint64_t terminal_publication_count,
    const std::uint64_t publication_command_count,
    const std::uint64_t window_state_count,
    const std::uint64_t window_descriptor_state_count,
    const std::uint64_t profile_step_count,
    const std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &result) noexcept {
  if (ops.plan_pipeline_structure == nullptr ||
      projection.occurrence_count == 0u ||
      projection.backend_dispatch_count == 0u ||
      projection.backend_step_occurrence_count == 0u ||
      projection.backend_step_description_count == 0u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  result.backend_dispatch_count = projection.backend_dispatch_count;
  result.host_transient_bytes = projection.host_transient_bytes;
  result.backend_reset_dispatch_count = projection.backend_reset_dispatch_count;
  result.backend_step_occurrence_count =
      projection.backend_step_occurrence_count;
  result.backend_step_description_count =
      projection.backend_step_description_count;
  result.backend_status_entry_count = projection.backend_status_entry_count;
  result.backend_window_dispatch_count =
      projection.backend_window_dispatch_count;
  result.backend_indirect_dispatch_count =
      projection.backend_indirect_dispatch_count;
  result.backend_window_state_count = window_state_count;
  result.backend_window_descriptor_state_count = window_descriptor_state_count;
  result.backend_status_source_count = projection.backend_status_source_count;
  result.backend_telemetry_count = projection.backend_telemetry_count;
  result.backend_status_command_count = projection.backend_status_command_count;
  result.backend_telemetry_command_count =
      projection.backend_telemetry_command_count;
  result.backend_parameter_bytes = projection.backend_parameter_bytes;
  result.backend_publication_count = publication_count;
  result.backend_terminal_publication_count = terminal_publication_count;
  result.backend_publication_command_count = publication_command_count;
  result.backend_command_binding_slot_upper =
      projection.backend_command_binding_slot_upper;
  result.backend_profile_step_count = profile_step_count;
  result.backend_profile_command_count = profile_command_count;
  return ops.plan_pipeline_structure(context, result);
}

[[nodiscard]] rund::AccelCheck plan_runtime_backend_structure(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::span<const BackendPublish> publications,
    const std::uint64_t profile_step_count,
    const std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &structure) noexcept {
  if (runs.empty() || runs.size() != recurrences.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const BackendOps *ops = nullptr;
  PreparedKernelPipelineReservation projection{};
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    PreparedKernelRouteReservation route{};
    if (item == nullptr || !item->ok || state == nullptr ||
        candidate == nullptr || candidate->plan_pipeline_private == nullptr ||
        (ops != nullptr && ops != candidate)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    bool duplicate_route = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate_route = true;
        break;
      }
    }
    ops = candidate;
    if (duplicate_route) {
      continue;
    }
    const rund::AccelCheck planned =
        candidate->plan_pipeline_private(state->bound.run, route);
    std::uint64_t entry_count = 0u;
    std::uint64_t occurrence_count = 0u;
    std::uint64_t window_count = 0u;
    std::uint64_t nested_group_count = 0u;
    std::uint64_t map_recurrence_group_count = 0u;
    std::uint64_t map_recurrence_history_group_count = 0u;
    std::uint64_t recurrence_hi = 0u;
    std::uint64_t recurrence_lo = 0u;
    if (!planned.ok) {
      return rund::AccelCheck{false, planned.reason == nullptr
                                         ? "compute_pipeline_capacity"
                                         : planned.reason};
    }
    if (!route_recurrence_shape(
            runs, recurrences, item->owner.get(), entry_count, occurrence_count,
            window_count, nested_group_count, map_recurrence_group_count,
            map_recurrence_history_group_count, recurrence_hi, recurrence_lo)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (!accumulate_route_projection(projection, route, entry_count,
                                     occurrence_count, window_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  const PreparedKernelPipelineShape shape =
      runtime_pipeline_shape(publications, recurrences, 1u, 1u, false);
  if (ops == nullptr ||
      projection.occurrence_count != structure.occurrence_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return finalize_pipeline_backend_structure(
      context, *ops, projection, publications.size(),
      shape.terminal_publication_count,
      shape.backend_publication_command_count, shape.window_state_count,
      shape.window_descriptor_state_count, profile_step_count,
      profile_command_count, structure);
}

[[nodiscard]] bool expand_pipeline(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const std::uint8_t> template_barriers,
    const std::span<const BackendPublish> publications,
    const std::span<const std::uint32_t> declared_steps,
    const std::uint32_t declared_step_count, const bool profile_steps,
    const std::uint32_t direct_aggregate_commands, ExpandedPipeline &expanded) {
  expanded.failure.stage(PreparedPipelineFailureStage::CommonExpansion);
  if (templates.empty() || templates.size() != template_barriers.size() ||
      templates.size() != declared_steps.size()) {
    return false;
  }
  std::vector<std::uint32_t> group_transducers;
  std::uint64_t command_count = 0u;
  for (std::size_t index = 0u; index < templates.size();) {
    expanded.failure.template_route(static_cast<std::uint32_t>(index));
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      ++command_count;
      ++index;
      continue;
    }
    std::size_t end = 0u;
    std::uint32_t outer_bound = 0u;
    std::uint32_t inner_bound = 0u;
    if (window->phase != BackendWindowPhase::NestedSeed ||
        !nested_shape(templates, index, end, outer_bound, inner_bound)) {
      return false;
    }
    NestedAggregate aggregate =
        BuildNestedAggregate(templates, template_barriers, publications, index);
    if (aggregate.invalid()) {
      expanded.reason = aggregate.reason;
      return false;
    }
    if (aggregate.ready()) {
      try {
        expanded.aggregates.push_back(std::move(aggregate));
      } catch (const std::bad_alloc &) {
        expanded.reason = "compute_pipeline_capacity";
        return false;
      }
    }
    const NestedAggregate *const direct = expanded.aggregates.size() == 1u
                                              ? &expanded.aggregates.front()
                                              : nullptr;
    bool declared_seed_range = direct != nullptr && direct->seed.first == 0u;
    if (declared_seed_range) {
      const std::uint32_t first = declared_steps[direct->seed.first];
      for (std::uint32_t outer = 0u; outer < direct->seed.count; ++outer) {
        if (first > std::numeric_limits<std::uint32_t>::max() - outer ||
            declared_steps[direct->seed.first + outer] != first + outer) {
          declared_seed_range = false;
          break;
        }
      }
    }
    bool profile_layout = !profile_steps;
    if (profile_steps && declared_step_count == templates.size()) {
      profile_layout = true;
      for (std::size_t declared = 0u; declared < declared_steps.size();
           ++declared) {
        if (declared_steps[declared] != declared) {
          profile_layout = false;
          break;
        }
      }
    }
    const bool complete_direct =
        direct_aggregate_commands != 0u && direct != nullptr && index == 0u &&
        end == templates.size() && direct->seed.count == outer_bound &&
        direct->action.first == direct->seed.end() &&
        direct->action.count == inner_bound &&
        direct->fold.first == direct->action.end() &&
        direct->fold.count == 3u && direct->fold.end() == templates.size() &&
        publications.size() == 1u && direct->publication_index == 0u &&
        direct->failure.logical_step == declared_steps.front() &&
        direct->profile.aggregate_profile_supported && declared_seed_range &&
        profile_layout;
    if (complete_direct) {
      // The native aggregate consumes compact templates directly. Retaining
      // K*(N+2) occurrence descriptors, barriers, and copied window records
      // would recreate the intermediate memory layer this proof eliminates.
      expanded.command_count = direct_aggregate_commands;
      expanded.compact_aggregate = true;
      return true;
    }
    if (group_transducers.empty()) {
      group_transducers.assign(templates.size(), NoTileTransducer);
    }
    const std::size_t action_first = index + outer_bound;
    MapRecurrence recurrence = BuildNestedMapRecurrence(
        templates.subspan(action_first, inner_bound),
        template_barriers.subspan(action_first, inner_bound));
    if (recurrence.invalid()) {
      expanded.reason = recurrence.reason;
      return false;
    }
    const bool fused = recurrence.ready();
    if (fused) {
      // Proof retains only the canonical Program pointer plus its exact source
      // recipe. No transformed artifact survives expansion, so nested group
      // count cannot recreate an intermediate source/metadata memory layer.
      if (recurrence.canonical_artifact == nullptr ||
          !recurrence.source_plan.ok || recurrence.history != nullptr) {
        expanded.reason = "accel_kernel_run_invalid";
        return false;
      }
      if (expanded.transducers.size() >= NoTileTransducer) {
        expanded.reason = "compute_pipeline_capacity";
        return false;
      }
      group_transducers[index] =
          static_cast<std::uint32_t>(expanded.transducers.size());
      expanded.transducers.push_back(TileTransducer{
          .recurrence = std::move(recurrence),
          .template_first = static_cast<std::uint32_t>(action_first),
          .template_count = inner_bound,
      });
    }
    const std::uint64_t commands_per_outer =
        fused ? 3u : static_cast<std::uint64_t>(inner_bound) + 2u;
    if (command_count > std::numeric_limits<std::uint32_t>::max() ||
        (commands_per_outer != 0u &&
         outer_bound >
             (std::numeric_limits<std::uint32_t>::max() - command_count) /
                 commands_per_outer)) {
      return false;
    }
    command_count +=
        static_cast<std::uint64_t>(outer_bound) * commands_per_outer;
    index = end;
  }
  if (command_count == 0u ||
      command_count > std::numeric_limits<std::uint32_t>::max() ||
      command_count > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  expanded.command_count = static_cast<std::uint32_t>(command_count);

  const std::size_t capacity = static_cast<std::size_t>(command_count);
  expanded.commands.reserve(capacity);
  expanded.barriers.reserve(capacity);
  expanded.windows.reserve(capacity);
  const auto append = [&](const std::size_t template_index,
                          const std::uint8_t barrier, const std::uint32_t outer,
                          const std::uint32_t outer_bound,
                          const std::uint32_t inner,
                          const std::uint32_t inner_bound,
                          const std::uint32_t route,
                          const std::uint32_t transducer = NoTileTransducer,
                          const std::uint32_t inner_advance =
                              NoTileTransducer) {
    if (template_index >= templates.size() ||
        expanded.commands.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    BackendBatchEntry command = templates[template_index];
    command.template_index = static_cast<std::uint32_t>(template_index);
    command.occurrence_index =
        static_cast<std::uint32_t>(expanded.commands.size());
    expanded.failure.occurrence_route(command);
    command.transducer = transducer;
    if (command.recurrence.window != nullptr) {
      expanded.windows.push_back(*command.recurrence.window);
      BackendWindow &window = expanded.windows.back();
      window.iteration = outer;
      window.bound = outer_bound;
      window.outer_iteration = outer;
      window.outer_bound = outer_bound;
      window.inner_iteration = inner;
      window.inner_bound = inner_bound;
      window.inner_advance =
          inner_advance != NoTileTransducer
              ? inner_advance
              : (window.phase == BackendWindowPhase::NestedAction ? 1u : 0u);
      window.route = route;
      command.recurrence.window = &window;
    }
    expanded.commands.push_back(std::move(command));
    expanded.barriers.push_back(barrier);
    return true;
  };

  for (std::size_t index = 0u; index < templates.size();) {
    expanded.failure.template_route(static_cast<std::uint32_t>(index));
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      const std::uint32_t iteration =
          window == nullptr ? 0u : window->outer_iteration;
      const std::uint32_t bound = window == nullptr ? 1u : window->outer_bound;
      if (!append(index, template_barriers[index], iteration, bound, 0u, 1u,
                  0u)) {
        return false;
      }
      ++index;
      continue;
    }

    std::size_t end = 0u;
    std::uint32_t outer_bound = 0u;
    std::uint32_t inner_bound = 0u;
    if (!nested_shape(templates, index, end, outer_bound, inner_bound)) {
      return false;
    }
    const std::size_t action_first = index + outer_bound;
    const std::size_t fold_first = action_first + inner_bound;
    const std::uint32_t transducer = group_transducers[index];
    for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
      const bool first_command = expanded.commands.empty();
      const std::uint8_t seed_barrier =
          first_command ? template_barriers[index + outer] : 1u;
      if (!append(index + outer, seed_barrier, outer, outer_bound, 0u,
                  inner_bound, 0u)) {
        return false;
      }
      if (transducer != NoTileTransducer) {
        if (!append(action_first, 1u, outer, outer_bound, 0u, inner_bound, 0u,
                    transducer, 0u)) {
          return false;
        }
      } else {
        for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
          if (!append(action_first + inner, 1u, outer, outer_bound, inner,
                      inner_bound, 0u)) {
            return false;
          }
        }
      }
      const std::uint32_t route =
          outer == 0u ? 0u : ((outer & 1u) != 0u ? 1u : 2u);
      if (!append(fold_first + route, 1u, outer, outer_bound, inner_bound,
                  inner_bound, route, NoTileTransducer,
                  transducer == NoTileTransducer ? 0u : inner_bound)) {
        return false;
      }
    }
    index = end;
  }
  return expanded.commands.size() == capacity &&
         expanded.barriers.size() == capacity;
}

[[nodiscard]] PreparedKernelPipeline
reject_pipeline(const PreparedPipelineFailureContext &context,
                const char *const reason) noexcept {
  return PreparedKernelPipeline{.failure = context.failure(reason)};
}

} // namespace

bool AccumulatePreparedKernelRouteProjectionForContract(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    const std::uint64_t compact_entry_count,
    const std::uint64_t occurrence_count,
    const std::uint64_t window_count) noexcept {
  return accumulate_route_projection(projection, route, compact_entry_count,
                                     occurrence_count, window_count);
}

bool BackendTemplateRouteDemandForContract(
    const std::uint64_t owner_count, const std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept {
  return backend_template_route_demand(owner_count, route_copies, demand);
}

bool PlanBackendTemplateRouteDemandsForContract(
    const std::span<const PreparedKernelRun *const> runs,
    const std::uint32_t route_copies,
    const std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept {
  return plan_backend_template_route_demands(
      runs, route_copies, demands, unique_route_count, template_count);
}

bool BackendPreparationCursorLifecycleForContract(
    BackendRun &run, const BackendTemplateRouteDemand demand) noexcept {
  if (run.templates != nullptr || run.failed_node != nullptr ||
      !run.template_route_demand.empty() || !demand.valid()) {
    return false;
  }
  PreparedKernelTemplateRegistry templates{};
  std::uint32_t failed_node = NoNode;
  const bool escaped = [&]() noexcept {
    const BackendPreparationCursor cursor{run, templates, failed_node, demand};
    if (run.templates != &templates || run.failed_node != &failed_node ||
        run.template_route_demand.owner_count != demand.owner_count ||
        run.template_route_demand.route_copies != demand.route_copies ||
        run.template_route_demand.capacity != demand.capacity) {
      return true;
    }
    // Exercise destruction across an early-return edge, matching every
    // fail-closed return from private backend materialization.
    return false;
  }();
  return !escaped && run.templates == nullptr && run.failed_node == nullptr &&
         run.template_route_demand.empty();
}

bool ScalePreparedMapRecurrenceRoutesForContract(
    const PreparedMapRecurrenceReservation &reservation,
    const std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept {
  return scale_map_recurrence_route_reservation(reservation, copies, scaled);
}

std::shared_ptr<void> FindPreparedKernelTemplate(
    const PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *const authority, const std::uint64_t variant_hi,
    const std::uint64_t variant_lo, const PreparedKernelTemplateMatch match,
    const void *const probe) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || authority == nullptr || match == nullptr ||
      probe == nullptr) {
    return {};
  }
  std::lock_guard lock{state->mutex};
  for (const PreparedKernelTemplateEntry &entry : state->entries) {
    if (entry.authority == authority && entry.variant_hi == variant_hi &&
        entry.variant_lo == variant_lo && entry.prepared != nullptr &&
        match(entry.prepared.get(), probe)) {
      return entry.prepared;
    }
  }
  return {};
}

rund::AccelCheck PublishPreparedKernelTemplate(
    PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *const authority, const std::uint64_t variant_hi,
    const std::uint64_t variant_lo, const BackendOps &ops,
    const PreparedKernelTemplateMatch match, const void *const probe,
    std::shared_ptr<void> &prepared) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || authority == nullptr || match == nullptr ||
      probe == nullptr || prepared == nullptr || ops.api != state->api ||
      ops.observe_pipeline_template == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  std::lock_guard lock{state->mutex};
  for (const PreparedKernelTemplateEntry &entry : state->entries) {
    if (entry.authority == authority && entry.variant_hi == variant_hi &&
        entry.variant_lo == variant_lo && entry.ops == &ops &&
        entry.prepared != nullptr && match(entry.prepared.get(), probe)) {
      prepared = entry.prepared;
      return rund::AccelCheck{true, "ok"};
    }
  }
  try {
    state->entries.push_back(PreparedKernelTemplateEntry{
        .authority = authority,
        .variant_hi = variant_hi,
        .variant_lo = variant_lo,
        .ops = &ops,
        .prepared = prepared,
    });
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck BindPreparedKernelTemplateRegistry(
    const rund::AccelApi api, const std::uint64_t context_id,
    PreparedKernelTemplateRegistry &registry) noexcept {
  if ((api != rund::AccelApi::Metal && api != rund::AccelApi::Vulkan) ||
      context_id == 0u) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  if (registry.owner == nullptr) {
    try {
      auto state = std::make_shared<PreparedKernelTemplateRegistryState>();
      state->context_id = context_id;
      state->api = api;
      if (registry.limit.ok) {
        if (registry.limit.template_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        const std::size_t template_count =
            static_cast<std::size_t>(registry.limit.template_count);
        state->entries.reserve(template_count);
        state->template_charges.reserve(template_count);
        // The cumulative transaction starts with the immutable public-plan
        // identity. Charges intentionally carry no fingerprint of their own;
        // accumulation therefore preserves this seed while primary and
        // alternate streams consume the same frozen limit.
        state->consumed.ok = true;
        state->consumed.reason = "ok";
        state->consumed.fingerprint_hi = registry.limit.fingerprint_hi;
        state->consumed.fingerprint_lo = registry.limit.fingerprint_lo;
        state->consumed.template_capacity = registry.limit.template_capacity;
        state->consumed.template_step_capacity =
            registry.limit.template_step_capacity;
        state->consumed.descriptor_set_capacity =
            registry.limit.descriptor_set_capacity;
        state->consumed.descriptor_capacity =
            registry.limit.descriptor_capacity;
      }
      registry.owner = std::move(state);
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || state->context_id != context_id ||
      state->api != api) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  if (registry.limit.ok &&
      (state->consumed.fingerprint_hi != registry.limit.fingerprint_hi ||
       state->consumed.fingerprint_lo != registry.limit.fingerprint_lo ||
       state->consumed.template_capacity != registry.limit.template_capacity ||
       state->consumed.template_step_capacity !=
           registry.limit.template_step_capacity ||
       state->consumed.descriptor_set_capacity !=
           registry.limit.descriptor_set_capacity ||
       state->consumed.descriptor_capacity !=
           registry.limit.descriptor_capacity)) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  registry.reservation = state->consumed;
  return rund::AccelCheck{true, "ok"};
}

bool PreparedKernelPipelineReservationWithin(
    const PreparedKernelPipelineReservation &reservation,
    const PreparedKernelPipelineReservation &limit) noexcept {
  return reservation.ok && limit.ok &&
         reservation.fingerprint_hi == limit.fingerprint_hi &&
         reservation.fingerprint_lo == limit.fingerprint_lo &&
         reservation.host_bytes <= limit.host_bytes &&
         reservation.native_bytes <= limit.native_bytes &&
         reservation.route_host_bytes <= limit.route_host_bytes &&
         reservation.route_native_bytes <= limit.route_native_bytes &&
         reservation.template_host_bytes <= limit.template_host_bytes &&
         reservation.template_native_bytes <= limit.template_native_bytes &&
         reservation.template_source_bytes <= limit.template_source_bytes &&
         reservation.source_transient_bytes <= limit.source_transient_bytes &&
         reservation.host_transient_bytes <= limit.host_transient_bytes &&
         reservation.route_count <= limit.route_count &&
         reservation.template_count <= limit.template_count &&
         reservation.template_step_count <= limit.template_step_count &&
         reservation.template_step_count <= limit.template_step_capacity &&
         reservation.template_step_capacity == limit.template_step_capacity &&
         reservation.route_step_count <= limit.route_step_count &&
         reservation.authored_entry_count <= limit.authored_entry_count &&
         reservation.occurrence_count <= limit.occurrence_count &&
         reservation.window_count <= limit.window_count &&
         reservation.nested_group_count <= limit.nested_group_count &&
         reservation.backend_dispatch_count <= limit.backend_dispatch_count &&
         reservation.backend_reset_dispatch_count <=
             limit.backend_reset_dispatch_count &&
         reservation.backend_window_dispatch_count <=
             limit.backend_window_dispatch_count &&
         reservation.backend_indirect_dispatch_count <=
             limit.backend_indirect_dispatch_count &&
         reservation.backend_window_state_count <=
             limit.backend_window_state_count &&
         reservation.backend_window_descriptor_state_count <=
             limit.backend_window_descriptor_state_count &&
         reservation.backend_step_occurrence_count <=
             limit.backend_step_occurrence_count &&
         reservation.backend_step_description_count <=
             limit.backend_step_description_count &&
         reservation.backend_status_source_count <=
             limit.backend_status_source_count &&
         reservation.backend_status_entry_count <=
             limit.backend_status_entry_count &&
         reservation.backend_telemetry_count <= limit.backend_telemetry_count &&
         reservation.backend_status_command_count <=
             limit.backend_status_command_count &&
         reservation.backend_telemetry_command_count <=
             limit.backend_telemetry_command_count &&
         reservation.backend_publication_count <=
             limit.backend_publication_count &&
         reservation.backend_terminal_publication_count <=
             limit.backend_terminal_publication_count &&
         reservation.backend_window_control_command_count <=
             limit.backend_window_control_command_count &&
         reservation.backend_publication_command_count <=
             limit.backend_publication_command_count &&
         reservation.backend_command_count <= limit.backend_command_count &&
         reservation.backend_command_chunk_count <=
             limit.backend_command_chunk_count &&
         reservation.backend_command_native_bytes <=
             limit.backend_command_native_bytes &&
         reservation.backend_command_binding_slot_upper <=
             limit.backend_command_binding_slot_upper &&
         reservation.backend_command_binding_count <=
             limit.backend_command_binding_count &&
         reservation.backend_parameter_bytes <= limit.backend_parameter_bytes &&
         reservation.backend_profile_step_count <=
             limit.backend_profile_step_count &&
         reservation.backend_profile_command_count <=
             limit.backend_profile_command_count &&
         reservation.backend_query_count <= limit.backend_query_count &&
         reservation.backend_native_buffer_count <=
             limit.backend_native_buffer_count &&
         reservation.backend_native_object_count <=
             limit.backend_native_object_count &&
         reservation.descriptor_set_count <= limit.descriptor_set_count &&
         reservation.descriptor_count <= limit.descriptor_count &&
         reservation.native_allocation_count <= limit.native_allocation_count &&
         reservation.map_recurrence.route_host_bytes <=
             limit.map_recurrence.route_host_bytes &&
         reservation.map_recurrence.route_native_bytes <=
             limit.map_recurrence.route_native_bytes &&
         reservation.map_recurrence.template_host_bytes <=
             limit.map_recurrence.template_host_bytes &&
         reservation.map_recurrence.template_native_bytes <=
             limit.map_recurrence.template_native_bytes &&
         reservation.map_recurrence.template_source_bytes <=
             limit.map_recurrence.template_source_bytes &&
         reservation.map_recurrence.source_transient_bytes <=
             limit.map_recurrence.source_transient_bytes &&
         reservation.map_recurrence.group_count <=
             limit.map_recurrence.group_count &&
         reservation.map_recurrence.history_group_count <=
             limit.map_recurrence.history_group_count &&
         reservation.map_recurrence.template_count <=
             limit.map_recurrence.template_count &&
         reservation.map_recurrence.terminal_template_group_capacity <=
             limit.map_recurrence.terminal_template_group_capacity &&
         reservation.map_recurrence.history_template_group_capacity <=
             limit.map_recurrence.history_template_group_capacity &&
         reservation.map_recurrence.route_step_count <=
             limit.map_recurrence.route_step_count &&
         reservation.map_recurrence.template_step_count <=
             limit.map_recurrence.template_step_count &&
         reservation.map_recurrence.descriptor_set_count <=
             limit.map_recurrence.descriptor_set_count &&
         reservation.map_recurrence.descriptor_count <=
             limit.map_recurrence.descriptor_count &&
         reservation.map_recurrence.route_native_allocation_count <=
             limit.map_recurrence.route_native_allocation_count &&
         reservation.map_recurrence.template_native_allocation_count <=
             limit.map_recurrence.template_native_allocation_count &&
         reservation.template_count <= limit.template_capacity &&
         reservation.descriptor_set_count <= limit.descriptor_set_capacity &&
         reservation.descriptor_count <= limit.descriptor_capacity;
}

namespace {

[[nodiscard]] bool
same_charged_template(const PreparedKernelTemplateCharge &charged,
                      const BackendOps &ops, const BackendRun &probe,
                      const PreparedKernelTemplateChargeKind kind) noexcept {
  return charged.ops == &ops && charged.kind == kind &&
         charged.probe != nullptr && ops.same_pipeline_template != nullptr &&
         ops.same_pipeline_template(probe, charged.probe->bound.run) &&
         ops.same_pipeline_template(charged.probe->bound.run, probe);
}

[[nodiscard]] bool same_charged_recurrence_template(
    const PreparedKernelTemplateCharge &charged, const BackendOps &ops,
    const MapRecurrencePreparationPlan &probe, const bool history) noexcept {
  const PreparedKernelTemplateChargeKind kind =
      history ? PreparedKernelTemplateChargeKind::RecurrenceHistory
              : PreparedKernelTemplateChargeKind::RecurrenceTerminal;
  if (charged.ops != &ops || charged.kind != kind || charged.probe == nullptr) {
    return false;
  }
  const MapRecurrencePreparationPlan cached = PlanMapRecurrencePreparation(
      charged.probe->bound.run, 1u, history ? 1u : 0u);
  return SameMapRecurrenceTemplate(probe, cached, history);
}

// Reserves the combined public-plan budget before backend/native
// materialization. Route state is charged for every stream. An immutable
// Program template is charged only by the first semantically equal stream in
// this registry. The registry mutex makes concurrent primary/alternate cold
// preparation one transaction with respect to the frozen limit.
[[nodiscard]] rund::AccelCheck
reserve_pipeline_budget(PreparedKernelTemplateRegistry &registry,
                        const std::span<const PreparedKernelRun *const> runs,
                        const std::span<const BackendRecurrence> recurrences,
                        const PreparedKernelPipelineReservation &structure,
                        const PreparedMapRecurrenceReservation &map_recurrence,
                        const std::uint32_t route_copies,
                        PipelineBudgetTransaction &transaction) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || runs.empty() || runs.size() != recurrences.size() ||
      !registry.limit.ok) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }

  std::array<RuntimeRecurrenceRoutePlan, PreparedPipelineStepCapacity>
      recurrence_routes{};
  if (!plan_runtime_recurrence_routes(
          runs, recurrences, route_copies,
          std::span<RuntimeRecurrenceRoutePlan>{recurrence_routes.data(),
                                                runs.size()})) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }

  try {
    if (!transaction.begin(*state, registry)) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    PreparedKernelPipelineReservation charge{};
    charge.ok = true;
    charge.reason = "ok";
    charge.template_capacity = registry.limit.template_capacity;
    charge.descriptor_set_capacity = registry.limit.descriptor_set_capacity;
    charge.descriptor_capacity = registry.limit.descriptor_capacity;
    charge.host_bytes = sizeof(prepared::PipelineState);
    std::uint64_t state_pointer_bytes = 0u;
    if (!multiply(runs.size(), sizeof(std::shared_ptr<prepared::RunState>),
                  state_pointer_bytes) ||
        !accumulate(charge.host_bytes, state_pointer_bytes) ||
        !accumulate_reservation(charge, structure)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const PreparedMapRecurrenceReservation recurrence_route =
        map_recurrence_route_reservation(map_recurrence);
    PreparedKernelPipelineReservation recurrence_route_charge{};
    if (!project_map_recurrence_reservation(recurrence_route,
                                            recurrence_route_charge) ||
        !accumulate_reservation(charge, recurrence_route_charge)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    PreparedMapRecurrenceReservation observed_recurrence_route{};
    PreparedMapRecurrenceReservation observed_recurrence_templates{};
    std::array<bool, PreparedPipelineStepCapacity> recurrence_terminal{};
    std::array<bool, PreparedPipelineStepCapacity> recurrence_history{};
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      const PreparedKernelRun *const item = runs[index];
      auto route =
          item == nullptr
              ? std::shared_ptr<prepared::RunState>{}
              : std::static_pointer_cast<prepared::RunState>(item->owner);
      const BackendOps *const ops =
          route == nullptr ? nullptr : route->bound.run.ops;
      if (item == nullptr || !item->ok || route == nullptr || ops == nullptr ||
          ops->plan_pipeline_private == nullptr ||
          ops->same_pipeline_template == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }

      bool duplicate_route = false;
      for (std::size_t prior = 0u; prior < index; ++prior) {
        const PreparedKernelRun *const previous = runs[prior];
        if (previous != nullptr && previous->owner.get() == item->owner.get()) {
          duplicate_route = true;
          break;
        }
      }
      if (duplicate_route) {
        continue;
      }

      PreparedKernelRouteReservation planned{};
      const rund::AccelCheck checked =
          ops->plan_pipeline_private(route->bound.run, planned);
      if (!checked.ok || planned.route_step_count == 0u ||
          planned.template_step_count == 0u ||
          planned.template_capacity == 0u ||
          !accumulate(planned.route_host_bytes, sizeof(prepared::RunState))) {
        return rund::AccelCheck{false, checked.reason == nullptr
                                           ? "compute_pipeline_capacity"
                                           : checked.reason};
      }
      PreparedKernelPipelineReservation route_charge{};
      route_charge.ok = true;
      route_charge.reason = "ok";
      route_charge.host_bytes = planned.route_host_bytes;
      route_charge.native_bytes = planned.route_native_bytes;
      route_charge.route_host_bytes = planned.route_host_bytes;
      route_charge.route_native_bytes = planned.route_native_bytes;
      route_charge.route_count = 1u;
      route_charge.route_step_count = planned.route_step_count;
      route_charge.descriptor_set_count = planned.descriptor_set_count;
      route_charge.descriptor_count = planned.descriptor_count;
      route_charge.native_allocation_count =
          planned.route_native_allocation_count;
      if (!accumulate_reservation(charge, route_charge)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }

      const RuntimeRecurrenceRoutePlan &recurrence_route_plan =
          recurrence_routes[index];
      if (!recurrence_route_plan.first_owner) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      MapRecurrencePreparationPlan recurrence_plan =
          PlanMapRecurrencePreparation(
              route->bound.run, recurrence_route_plan.group_count,
              recurrence_route_plan.history_group_count);
      recurrence_plan.terminal_template_group_capacity =
          recurrence_plan.terminal_group_count() == 0u
              ? 0u
              : recurrence_route_plan.template_capacities.terminal;
      recurrence_plan.history_template_group_capacity =
          recurrence_plan.history_group_count == 0u
              ? 0u
              : recurrence_route_plan.template_capacities.history;
      recurrence_terminal[index] = recurrence_plan.terminal_group_count() != 0u;
      recurrence_history[index] = recurrence_plan.history_group_count != 0u;
      PreparedMapRecurrenceReservation recurrence{};
      const rund::AccelCheck recurrence_checked =
          plan_map_recurrence_reservation(*ops, recurrence_plan, recurrence);
      PreparedMapRecurrenceReservation terminal_reservation{};
      PreparedMapRecurrenceReservation history_reservation{};
      const rund::AccelCheck recurrence_variants_checked =
          recurrence_checked.ok ? verify_map_recurrence_template_variants(
                                      *ops, recurrence_plan, recurrence,
                                      terminal_reservation, history_reservation)
                                : recurrence_checked;
      if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
          !accumulate_map_recurrence_route(
              observed_recurrence_route,
              map_recurrence_route_reservation(recurrence))) {
        return rund::AccelCheck{
            false, !recurrence_checked.ok
                       ? (recurrence_checked.reason == nullptr
                              ? "compute_pipeline_capacity"
                              : recurrence_checked.reason)
                       : (recurrence_variants_checked.reason == nullptr
                              ? "compute_pipeline_capacity"
                              : recurrence_variants_checked.reason)};
      }

      const auto current_recurrence_template_seen =
          [&](const bool history) noexcept {
            for (std::size_t prior = 0u; prior < index; ++prior) {
              const PreparedKernelRun *const previous = runs[prior];
              const auto *const previous_state =
                  previous == nullptr ? nullptr
                                      : static_cast<const prepared::RunState *>(
                                            previous->owner.get());
              if (!(history ? recurrence_history[prior]
                            : recurrence_terminal[prior]) ||
                  previous_state == nullptr) {
                continue;
              }
              const RuntimeRecurrenceRoutePlan &previous_route_plan =
                  recurrence_routes[prior];
              const MapRecurrencePreparationPlan previous_plan =
                  PlanMapRecurrencePreparation(
                      previous_state->bound.run,
                      previous_route_plan.group_count,
                      previous_route_plan.history_group_count);
              if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan,
                                            history)) {
                return true;
              }
            }
            return false;
          };
      const auto charge_recurrence_template =
          [&](const bool present, const bool history,
              const PreparedMapRecurrenceReservation &reservation) {
            if (!present || current_recurrence_template_seen(history)) {
              return true;
            }
            if (!accumulate_map_recurrence_template(
                    observed_recurrence_templates, reservation)) {
              return false;
            }
            const PreparedKernelTemplateChargeKind kind =
                history ? PreparedKernelTemplateChargeKind::RecurrenceHistory
                        : PreparedKernelTemplateChargeKind::RecurrenceTerminal;
            bool recurrence_template_seen = false;
            for (const PreparedKernelTemplateCharge &charged :
                 state->template_charges) {
              if (same_charged_recurrence_template(charged, *ops,
                                                   recurrence_plan, history)) {
                recurrence_template_seen = true;
                break;
              }
            }
            if (recurrence_template_seen) {
              return true;
            }
            PreparedKernelPipelineReservation recurrence_template_charge{};
            if (!project_map_recurrence_reservation(
                    reservation, recurrence_template_charge) ||
                !accumulate_reservation(charge, recurrence_template_charge) ||
                state->template_charges.size() ==
                    state->template_charges.capacity()) {
              return false;
            }
            state->template_charges.push_back(
                PreparedKernelTemplateCharge{route, ops, kind});
            return true;
          };
      if (!charge_recurrence_template(recurrence_terminal[index], false,
                                      terminal_reservation) ||
          !charge_recurrence_template(recurrence_history[index], true,
                                      history_reservation)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }

      bool template_seen = false;
      for (const PreparedKernelTemplateCharge &charged :
           state->template_charges) {
        if (same_charged_template(charged, *ops, route->bound.run,
                                  PreparedKernelTemplateChargeKind::Program)) {
          template_seen = true;
          break;
        }
      }
      if (template_seen) {
        continue;
      }

      PreparedKernelPipelineReservation template_charge{};
      template_charge.ok = true;
      template_charge.reason = "ok";
      template_charge.host_bytes = planned.template_host_bytes;
      template_charge.native_bytes = planned.template_native_bytes;
      template_charge.template_host_bytes = planned.template_host_bytes;
      template_charge.template_native_bytes = planned.template_native_bytes;
      template_charge.template_source_bytes = planned.template_source_bytes;
      template_charge.source_transient_bytes = planned.source_transient_bytes;
      template_charge.template_count = 1u;
      template_charge.template_step_count = planned.template_step_count;
      template_charge.native_allocation_count =
          planned.template_native_allocation_count;
      if (!accumulate_reservation(charge, template_charge)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      if (state->template_charges.size() ==
          state->template_charges.capacity()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      state->template_charges.push_back(PreparedKernelTemplateCharge{
          route, ops, PreparedKernelTemplateChargeKind::Program});
    }

    if (!same_map_recurrence_reservation(
            observed_recurrence_route,
            map_recurrence_route_reservation(map_recurrence)) ||
        !same_map_recurrence_reservation(
            observed_recurrence_templates,
            map_recurrence_template_reservation(map_recurrence))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    // Source specialization is serialized by the registry transaction. Only
    // the increase over the already consumed high-water can coexist with the
    // retained owners; primary/alternate streams must not add the same
    // transient a second time.
    if (charge.source_transient_bytes >
            state->consumed.source_transient_bytes &&
        !accumulate(charge.host_bytes,
                    charge.source_transient_bytes -
                        state->consumed.source_transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    // Backend cold-finalizer workspace is independent of source emission and
    // is likewise serialized per prepared stream. Charge only a new
    // high-water; no transient allocation becomes a retained registry owner.
    if (charge.host_transient_bytes > state->consumed.host_transient_bytes &&
        !accumulate(charge.host_bytes,
                    charge.host_transient_bytes -
                        state->consumed.host_transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    PreparedKernelPipelineReservation prospective = state->consumed;
    prospective.ok = true;
    prospective.reason = "ok";
    prospective.template_capacity = registry.limit.template_capacity;
    prospective.template_step_capacity = registry.limit.template_step_capacity;
    prospective.descriptor_set_capacity =
        registry.limit.descriptor_set_capacity;
    prospective.descriptor_capacity = registry.limit.descriptor_capacity;
    if (!accumulate_reservation(prospective, charge) ||
        !PreparedKernelPipelineReservationWithin(prospective, registry.limit)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    state->consumed = prospective;
    registry.reservation = prospective;
    return rund::AccelCheck{true, "ok"};
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
}

} // namespace

PreparedKernelPipelineReservation PlanPreparedKernelPipelineLimit(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelProgramRoute> routes,
    const PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry &templates) noexcept {
  PreparedKernelPipelineReservation result{};
  result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  if (!context.check.ok || context.id == 0u || routes.empty() ||
      routes.size() > PreparedPipelineStepCapacity ||
      shape.publication_count > 32u || shape.declared_step_count == 0u ||
      shape.declared_step_count > PreparedPipelineStepCapacity ||
      !valid_publication_shape(shape) ||
      shape.window_descriptor_state_count > shape.window_state_count ||
      ((shape.window_state_count == 0u) !=
       (shape.window_descriptor_state_count == 0u)) ||
      (shape.route_copies != 1u && shape.route_copies != 2u) ||
      (context.api != rund::AccelApi::Metal &&
       context.api != rund::AccelApi::Vulkan)) {
    result.reason = "accel_kernel_run_invalid";
    templates.limit = result;
    return result;
  }
  fingerprint_pipeline_header(result.fingerprint_hi, result.fingerprint_lo,
                              context, shape, routes.size());

  const BackendOps *ops = nullptr;
  std::uint64_t entry_count = 0u;
  std::uint64_t stream_count = 0u;
  PreparedKernelPipelineReservation backend_projection{};
  for (std::size_t route_index = 0u; route_index < routes.size();
       ++route_index) {
    const PreparedKernelProgramRoute &route = routes[route_index];
    if (route.kernel == nullptr || route.tile_count == 0u ||
        route.entry_count == 0u ||
        route.window_count > route.occurrence_count ||
        route.nested_group_count > route.entry_count ||
        route.map_recurrence_history_group_count >
            route.map_recurrence_group_count ||
        route.route_copies == 0u || route.route_copies > 2u ||
        route.route_copies != shape.route_copies) {
      result.reason = "accel_kernel_run_invalid";
      templates.limit = result;
      return result;
    }
    const KernelExecution execution =
        AdmitKernelForExecution(context, *route.kernel);
    const BackendOps *const candidate =
        execution.context_admission.pick == nullptr
            ? nullptr
            : execution.context_admission.pick->ops;
    if (!execution.admission.check.ok || candidate == nullptr ||
        candidate->api != context.api ||
        candidate->plan_pipeline_program == nullptr ||
        candidate->same_pipeline_program_template == nullptr ||
        (ops != nullptr && ops != candidate)) {
      result.reason = execution.admission.check.reason == nullptr
                          ? "accel_kernel_run_invalid"
                          : execution.admission.check.reason;
      templates.limit = result;
      return result;
    }
    ops = candidate;
    stream_count = std::max<std::uint64_t>(stream_count, route.route_copies);
    std::uint64_t entries = 0u;
    if (!multiply(route.entry_count, route.route_copies, entries) ||
        !accumulate(entry_count, entries)) {
      templates.limit = result;
      return result;
    }
    fingerprint_route(result.fingerprint_hi, result.fingerprint_lo,
                      route.kernel->kernel_id, route.kernel->graph_id_hi,
                      route.kernel->graph_id_lo, route.kernel->node_count,
                      route.kernel->api, route.tile_count, route.views,
                      route.scratch, route.entry_count, route.occurrence_count,
                      route.window_count, route.nested_group_count,
                      route.map_recurrence_group_count,
                      route.map_recurrence_history_group_count,
                      route.recurrence_fingerprint_hi,
                      route.recurrence_fingerprint_lo, route.route_copies);
    const backend_template_plan::MapSpecializationFingerprint map_fingerprint =
        backend_template_plan::program_map_specialization_fingerprint(execution,
                                                                      route);
    if (!map_fingerprint.ok) {
      result.reason = "accel_kernel_run_invalid";
      templates.limit = result;
      return result;
    }
    fingerprint_mix(result.fingerprint_hi, map_fingerprint.hi);
    fingerprint_mix(result.fingerprint_lo, map_fingerprint.lo);

    PreparedKernelRouteReservation planned{};
    const rund::AccelCheck checked =
        candidate->plan_pipeline_program(execution, route, planned);
    if (!checked.ok || planned.route_step_count == 0u ||
        planned.template_step_count == 0u || planned.template_capacity == 0u ||
        !accumulate(planned.route_host_bytes, sizeof(prepared::RunState))) {
      result.reason = checked.reason == nullptr ? "compute_pipeline_capacity"
                                                : checked.reason;
      templates.limit = result;
      return result;
    }
    if (!accumulate_route_projection(backend_projection, planned,
                                     route.entry_count, route.occurrence_count,
                                     route.window_count)) {
      result.reason = "compute_pipeline_capacity";
      templates.limit = result;
      return result;
    }
    result.template_capacity =
        std::min(result.template_capacity, planned.template_capacity);
    result.template_step_capacity =
        std::min(result.template_step_capacity, planned.template_step_capacity);
    std::uint64_t route_host = 0u;
    std::uint64_t route_native = 0u;
    std::uint64_t route_steps = 0u;
    std::uint64_t descriptor_sets = 0u;
    std::uint64_t descriptors = 0u;
    std::uint64_t route_allocations = 0u;
    if (!multiply(planned.route_host_bytes, route.route_copies, route_host) ||
        !multiply(planned.route_native_bytes, route.route_copies,
                  route_native) ||
        !multiply(planned.route_step_count, route.route_copies, route_steps) ||
        !multiply(planned.descriptor_set_count, route.route_copies,
                  descriptor_sets) ||
        !multiply(planned.descriptor_count, route.route_copies, descriptors) ||
        !multiply(planned.route_native_allocation_count, route.route_copies,
                  route_allocations) ||
        !accumulate(result.route_count, route.route_copies) ||
        !multiply(route.entry_count, route.route_copies, entries) ||
        !accumulate(result.authored_entry_count, entries) ||
        !multiply(route.occurrence_count, route.route_copies, entries) ||
        !accumulate(result.occurrence_count, entries) ||
        !multiply(route.window_count, route.route_copies, entries) ||
        !accumulate(result.window_count, entries) ||
        !multiply(route.nested_group_count, route.route_copies, entries) ||
        !accumulate(result.nested_group_count, entries) ||
        !accumulate(result.route_host_bytes, route_host) ||
        !accumulate(result.route_native_bytes, route_native) ||
        !accumulate(result.route_step_count, route_steps) ||
        !accumulate(result.descriptor_set_count, descriptor_sets) ||
        !accumulate(result.descriptor_count, descriptors) ||
        !accumulate(result.native_allocation_count, route_allocations)) {
      templates.limit = result;
      return result;
    }

    MapRecurrencePreparationPlan recurrence_plan =
        PlanMapRecurrencePreparation(execution, route);
    MapRecurrenceTemplateCapacities recurrence_template_capacities{};
    if (!recurrence_plan.ok ||
        !program_recurrence_template_capacities(
            execution, routes, route_index, recurrence_plan,
            recurrence_template_capacities)) {
      result.reason = recurrence_plan.reason == nullptr
                          ? "compute_pipeline_capacity"
                          : recurrence_plan.reason;
      templates.limit = result;
      return result;
    }
    recurrence_plan.terminal_template_group_capacity =
        recurrence_plan.terminal_group_count() == 0u
            ? 0u
            : recurrence_template_capacities.terminal;
    recurrence_plan.history_template_group_capacity =
        recurrence_plan.history_group_count == 0u
            ? 0u
            : recurrence_template_capacities.history;
    PreparedMapRecurrenceReservation recurrence{};
    const rund::AccelCheck recurrence_checked = plan_map_recurrence_reservation(
        *candidate, recurrence_plan, recurrence);
    PreparedMapRecurrenceReservation terminal_recurrence{};
    PreparedMapRecurrenceReservation history_recurrence{};
    const rund::AccelCheck recurrence_variants_checked =
        recurrence_checked.ok ? verify_map_recurrence_template_variants(
                                    *candidate, recurrence_plan, recurrence,
                                    terminal_recurrence, history_recurrence)
                              : recurrence_checked;
    PreparedMapRecurrenceReservation scaled_recurrence_route{};
    PreparedKernelPipelineReservation recurrence_route_projection{};
    if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
        !scale_map_recurrence_route_reservation(recurrence, route.route_copies,
                                                scaled_recurrence_route) ||
        !project_map_recurrence_reservation(scaled_recurrence_route,
                                            recurrence_route_projection) ||
        !accumulate_reservation(result, recurrence_route_projection)) {
      result.reason = !recurrence_checked.ok
                          ? (recurrence_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_checked.reason)
                          : (recurrence_variants_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_variants_checked.reason);
      templates.limit = result;
      return result;
    }
    const auto recurrence_variant_seen = [&](const bool history) noexcept {
      for (std::size_t prior = 0u; prior < route_index; ++prior) {
        const PreparedKernelProgramRoute &previous = routes[prior];
        if (!same_program_recurrence_authority(route, previous)) {
          continue;
        }
        const MapRecurrencePreparationPlan previous_plan =
            PlanMapRecurrencePreparation(execution, previous);
        if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan,
                                      history)) {
          return true;
        }
      }
      return false;
    };
    const auto accumulate_recurrence_template =
        [&](const bool present, const bool history,
            const PreparedMapRecurrenceReservation &reservation) {
          if (!present || recurrence_variant_seen(history)) {
            return true;
          }
          PreparedKernelPipelineReservation recurrence_template_projection{};
          return project_map_recurrence_reservation(
                     reservation, recurrence_template_projection) &&
                 accumulate_reservation(result, recurrence_template_projection);
        };
    if (!accumulate_recurrence_template(
            recurrence_plan.terminal_group_count() != 0u, false,
            terminal_recurrence) ||
        !accumulate_recurrence_template(recurrence_plan.history_group_count !=
                                            0u,
                                        true, history_recurrence)) {
      result.reason = "compute_pipeline_capacity";
      templates.limit = result;
      return result;
    }
    bool template_seen = false;
    for (std::size_t prior = 0u; prior < route_index; ++prior) {
      if (candidate->same_pipeline_program_template(execution, route,
                                                    routes[prior])) {
        template_seen = true;
        break;
      }
    }
    if (template_seen) {
      continue;
    }
    if (!accumulate(result.template_count, 1u) ||
        !accumulate(result.template_host_bytes, planned.template_host_bytes) ||
        !accumulate(result.template_native_bytes,
                    planned.template_native_bytes) ||
        !accumulate(result.template_source_bytes,
                    planned.template_source_bytes) ||
        !accumulate(result.template_step_count, planned.template_step_count) ||
        !accumulate(result.native_allocation_count,
                    planned.template_native_allocation_count)) {
      templates.limit = result;
      return result;
    }
    result.source_transient_bytes =
        std::max(result.source_transient_bytes, planned.source_transient_bytes);
  }

  const PreparedKernelPipelineReservation structure =
      plan_pipeline_structure_counts(
          result.authored_entry_count, result.occurrence_count,
          result.window_count, result.nested_group_count);
  PreparedKernelPipelineReservation backend_structure{};
  PreparedKernelPipelineReservation per_stream_backend{};
  const bool divisible_stream_shape =
      stream_count != 0u && structure.occurrence_count % stream_count == 0u &&
      structure.window_count % stream_count == 0u &&
      structure.nested_group_count % stream_count == 0u;
  const std::uint64_t per_stream_occurrences =
      divisible_stream_shape ? structure.occurrence_count / stream_count : 0u;
  if (divisible_stream_shape) {
    per_stream_backend.occurrence_count = per_stream_occurrences;
    per_stream_backend.window_count = structure.window_count / stream_count;
    per_stream_backend.nested_group_count =
        structure.nested_group_count / stream_count;
  }
  const bool backend_projection_matches =
      structure.ok && ops != nullptr && divisible_stream_shape &&
      backend_projection.occurrence_count == per_stream_occurrences;
  const rund::AccelCheck backend_planned =
      backend_projection_matches
          ? finalize_pipeline_backend_structure(
                context, *ops, backend_projection, shape.publication_count,
                shape.terminal_publication_count,
                shape.backend_publication_command_count,
                shape.window_state_count, shape.window_descriptor_state_count,
                shape.profile_steps ? shape.declared_step_count : 0u,
                shape.profile_steps ? per_stream_occurrences : 0u,
                per_stream_backend)
          : rund::AccelCheck{false, "compute_pipeline_capacity"};
  if (backend_planned.ok) {
    // These dimensions are already owned by the common expanded structure;
    // the backend consumes them only to size its private control owners.
    per_stream_backend.occurrence_count = 0u;
    per_stream_backend.window_count = 0u;
    per_stream_backend.nested_group_count = 0u;
    for (std::uint64_t stream = 0u; stream < stream_count; ++stream) {
      if (!accumulate_reservation(backend_structure, per_stream_backend)) {
        result.reason = "compute_pipeline_capacity";
        templates.limit = result;
        return result;
      }
    }
  }
  std::uint64_t pipeline_headers = 0u;
  std::uint64_t state_pointers = 0u;
  std::uint64_t registry_bytes = sizeof(PreparedKernelTemplateRegistryState);
  std::uint64_t registry_entries = 0u;
  std::uint64_t registry_charges = 0u;
  if (!structure.ok || !backend_planned.ok || ops == nullptr ||
      result.route_count == 0u || result.template_count == 0u ||
      result.template_count > result.template_capacity ||
      !multiply(stream_count, sizeof(prepared::PipelineState),
                pipeline_headers) ||
      !multiply(entry_count, sizeof(std::shared_ptr<prepared::RunState>),
                state_pointers) ||
      !multiply(result.template_count, sizeof(PreparedKernelTemplateEntry),
                registry_entries) ||
      !multiply(result.template_count, sizeof(PreparedKernelTemplateCharge),
                registry_charges) ||
      !accumulate(registry_bytes, registry_entries) ||
      !accumulate(registry_bytes, registry_charges) ||
      !add(pipeline_headers, state_pointers, result.host_bytes) ||
      !add(result.route_native_bytes, result.template_native_bytes,
           result.native_bytes) ||
      !accumulate(result.host_bytes, registry_bytes) ||
      !accumulate(result.host_bytes, structure.host_bytes) ||
      !accumulate_reservation(result, backend_structure) ||
      !accumulate(result.host_bytes, result.route_host_bytes) ||
      !accumulate(result.host_bytes, result.template_host_bytes) ||
      !accumulate(result.host_bytes, result.source_transient_bytes) ||
      !accumulate(result.host_bytes, result.host_transient_bytes) ||
      result.descriptor_set_count > result.descriptor_set_capacity ||
      result.descriptor_count > result.descriptor_capacity ||
      result.host_bytes > std::numeric_limits<std::size_t>::max()) {
    templates.limit = result;
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  templates.limit = result;
  return result;
}

PreparedKernelPipelineReservation PlanPreparedKernelPipeline(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry *const templates) noexcept {
  (void)templates;
  PreparedKernelPipelineReservation result{};
  if (runs.empty() || runs.size() != recurrences.size() ||
      runs.size() > PreparedPipelineStepCapacity ||
      shape.publication_count > 32u || shape.declared_step_count == 0u ||
      shape.declared_step_count > PreparedPipelineStepCapacity ||
      !valid_publication_shape(shape) ||
      (shape.route_copies != 1u && shape.route_copies != 2u)) {
    result.reason = "accel_kernel_run_invalid";
    return result;
  }

  std::uint64_t unique_route_count = 0u;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    if (item == nullptr || item->owner == nullptr) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    bool duplicate = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      duplicate = duplicate || (runs[prior] != nullptr &&
                                runs[prior]->owner.get() == item->owner.get());
    }
    unique_route_count += duplicate ? 0u : 1u;
  }
  fingerprint_pipeline_header(result.fingerprint_hi, result.fingerprint_lo,
                              context, shape, unique_route_count);

  std::array<RuntimeRecurrenceRoutePlan, PreparedPipelineStepCapacity>
      recurrence_routes{};
  if (!plan_runtime_recurrence_routes(
          runs, recurrences, shape.route_copies,
          std::span<RuntimeRecurrenceRoutePlan>{recurrence_routes.data(),
                                                runs.size()})) {
    result.reason = "accel_kernel_run_invalid";
    return result;
  }

  const BackendOps *ops = nullptr;
  std::uint64_t common_host_bytes = sizeof(prepared::PipelineState);
  std::uint64_t state_pointer_bytes = 0u;
  if (!multiply(runs.size(), sizeof(std::shared_ptr<prepared::RunState>),
                state_pointer_bytes) ||
      !accumulate(common_host_bytes, state_pointer_bytes)) {
    return result;
  }
  result.host_bytes = common_host_bytes;
  result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  std::array<bool, PreparedPipelineStepCapacity> recurrence_terminal{};
  std::array<bool, PreparedPipelineStepCapacity> recurrence_history{};

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr ||
        !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
        candidate->plan_pipeline_private == nullptr ||
        candidate->same_pipeline_template == nullptr ||
        !prepared::MatchesContext(context, *state) ||
        (ops != nullptr && ops != candidate)) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    ops = candidate;

    const RuntimeRecurrenceRoutePlan &recurrence_route_plan =
        recurrence_routes[index];
    if (!recurrence_route_plan.first_owner) {
      continue;
    }
    const KernelAdmission &admission = state->execution.admission;
    fingerprint_route(
        result.fingerprint_hi, result.fingerprint_lo, admission.kernel_id,
        admission.graph_id_hi, admission.graph_id_lo, admission.node_count,
        admission.api, state->tile_count, state->bound.run.views,
        state->bound.run.scratch, recurrence_route_plan.entry_count,
        recurrence_route_plan.occurrence_count,
        recurrence_route_plan.window_count,
        recurrence_route_plan.nested_group_count,
        recurrence_route_plan.group_count,
        recurrence_route_plan.history_group_count,
        recurrence_route_plan.fingerprint_hi,
        recurrence_route_plan.fingerprint_lo, shape.route_copies);
    const backend_template_plan::MapSpecializationFingerprint map_fingerprint =
        backend_template_plan::runtime_map_specialization_fingerprint(
            state->bound.run);
    if (!map_fingerprint.ok) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    fingerprint_mix(result.fingerprint_hi, map_fingerprint.hi);
    fingerprint_mix(result.fingerprint_lo, map_fingerprint.lo);
    if (!accumulate(result.route_count, 1u)) {
      return result;
    }

    PreparedKernelRouteReservation route{};
    const rund::AccelCheck planned =
        candidate->plan_pipeline_private(state->bound.run, route);
    if (!planned.ok || route.route_step_count == 0u ||
        route.template_step_count == 0u || route.template_capacity == 0u ||
        !accumulate(route.route_host_bytes, sizeof(prepared::RunState))) {
      result.reason = planned.reason == nullptr ? "compute_pipeline_capacity"
                                                : planned.reason;
      return result;
    }
    result.template_capacity =
        std::min(result.template_capacity, route.template_capacity);
    result.template_step_capacity =
        std::min(result.template_step_capacity, route.template_step_capacity);
    if (!accumulate(result.route_host_bytes, route.route_host_bytes) ||
        !accumulate(result.route_native_bytes, route.route_native_bytes) ||
        !accumulate(result.host_bytes, route.route_host_bytes) ||
        !accumulate(result.native_bytes, route.route_native_bytes) ||
        !accumulate(result.route_step_count, route.route_step_count) ||
        !accumulate(result.descriptor_set_count, route.descriptor_set_count) ||
        !accumulate(result.descriptor_count, route.descriptor_count) ||
        !accumulate(result.native_allocation_count,
                    route.route_native_allocation_count)) {
      return result;
    }

    MapRecurrencePreparationPlan recurrence_plan = PlanMapRecurrencePreparation(
        state->bound.run, recurrence_route_plan.group_count,
        recurrence_route_plan.history_group_count);
    recurrence_plan.terminal_template_group_capacity =
        recurrence_plan.terminal_group_count() == 0u
            ? 0u
            : recurrence_route_plan.template_capacities.terminal;
    recurrence_plan.history_template_group_capacity =
        recurrence_plan.history_group_count == 0u
            ? 0u
            : recurrence_route_plan.template_capacities.history;
    recurrence_terminal[index] = recurrence_plan.terminal_group_count() != 0u;
    recurrence_history[index] = recurrence_plan.history_group_count != 0u;
    PreparedMapRecurrenceReservation recurrence{};
    const rund::AccelCheck recurrence_checked = plan_map_recurrence_reservation(
        *candidate, recurrence_plan, recurrence);
    PreparedMapRecurrenceReservation terminal_recurrence{};
    PreparedMapRecurrenceReservation history_recurrence{};
    const rund::AccelCheck recurrence_variants_checked =
        recurrence_checked.ok ? verify_map_recurrence_template_variants(
                                    *candidate, recurrence_plan, recurrence,
                                    terminal_recurrence, history_recurrence)
                              : recurrence_checked;
    const PreparedMapRecurrenceReservation recurrence_route =
        map_recurrence_route_reservation(recurrence);
    PreparedKernelPipelineReservation recurrence_route_projection{};
    if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
        !project_map_recurrence_reservation(recurrence_route,
                                            recurrence_route_projection) ||
        !accumulate_reservation(result, recurrence_route_projection)) {
      result.reason = !recurrence_checked.ok
                          ? (recurrence_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_checked.reason)
                          : (recurrence_variants_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_variants_checked.reason);
      return result;
    }
    const auto recurrence_variant_seen = [&](const bool history) noexcept {
      for (std::size_t prior = 0u; prior < index; ++prior) {
        const PreparedKernelRun *const previous = runs[prior];
        const auto *const previous_state =
            previous == nullptr ? nullptr
                                : static_cast<const prepared::RunState *>(
                                      previous->owner.get());
        if (!(history ? recurrence_history[prior]
                      : recurrence_terminal[prior]) ||
            previous_state == nullptr) {
          continue;
        }
        const RuntimeRecurrenceRoutePlan &previous_route_plan =
            recurrence_routes[prior];
        const MapRecurrencePreparationPlan previous_plan =
            PlanMapRecurrencePreparation(
                previous_state->bound.run, previous_route_plan.group_count,
                previous_route_plan.history_group_count);
        if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan,
                                      history)) {
          return true;
        }
      }
      return false;
    };
    const auto accumulate_recurrence_template =
        [&](const bool present, const bool history,
            const PreparedMapRecurrenceReservation &reservation) {
          if (!present || recurrence_variant_seen(history)) {
            return true;
          }
          PreparedKernelPipelineReservation recurrence_template_projection{};
          return project_map_recurrence_reservation(
                     reservation, recurrence_template_projection) &&
                 accumulate_reservation(result, recurrence_template_projection);
        };
    if (!accumulate_recurrence_template(recurrence_terminal[index], false,
                                        terminal_recurrence) ||
        !accumulate_recurrence_template(recurrence_history[index], true,
                                        history_recurrence)) {
      result.reason = "compute_pipeline_capacity";
      return result;
    }

    bool template_seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      if (previous_state != nullptr &&
          candidate->same_pipeline_template(state->bound.run,
                                            previous_state->bound.run)) {
        template_seen = true;
        break;
      }
    }
    if (template_seen) {
      continue;
    }
    if (!accumulate(result.template_count, 1u) ||
        !accumulate(result.template_host_bytes, route.template_host_bytes) ||
        !accumulate(result.template_native_bytes,
                    route.template_native_bytes) ||
        !accumulate(result.host_bytes, route.template_host_bytes) ||
        !accumulate(result.native_bytes, route.template_native_bytes) ||
        !accumulate(result.template_source_bytes,
                    route.template_source_bytes) ||
        !accumulate(result.template_step_count, route.template_step_count) ||
        !accumulate(result.native_allocation_count,
                    route.template_native_allocation_count)) {
      return result;
    }
    result.source_transient_bytes =
        std::max(result.source_transient_bytes, route.source_transient_bytes);
  }

  if (!accumulate(result.host_bytes, result.source_transient_bytes) ||
      ops == nullptr || result.template_count == 0u ||
      result.template_count > result.template_capacity ||
      result.descriptor_set_count > result.descriptor_set_capacity ||
      result.descriptor_count > result.descriptor_capacity ||
      result.host_bytes > std::numeric_limits<std::size_t>::max()) {
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  return result;
}

[[nodiscard]] bool locate_template_step_capacity_failure(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::uint64_t capacity,
    PreparedPipelineFailureContext &failure) noexcept {
  std::uint64_t admitted_steps = 0u;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    if (state == nullptr || state->bound.run.ops == nullptr ||
        state->bound.run.ops->same_pipeline_template == nullptr) {
      return false;
    }
    bool duplicate_route = false;
    bool template_seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      duplicate_route =
          duplicate_route || (previous != nullptr && item != nullptr &&
                              previous->owner.get() == item->owner.get());
      template_seen =
          template_seen || (previous_state != nullptr &&
                            state->bound.run.ops->same_pipeline_template(
                                state->bound.run, previous_state->bound.run));
    }
    if (duplicate_route || template_seen) {
      continue;
    }
    const std::uint64_t step_count = state->bound.run.step_count;
    if (admitted_steps > capacity || step_count > capacity - admitted_steps) {
      const std::uint64_t crossing =
          admitted_steps >= capacity ? 0u : capacity - admitted_steps;
      const std::size_t step_index =
          crossing < step_count ? static_cast<std::size_t>(crossing) : 0u;
      if (index <= std::numeric_limits<std::uint32_t>::max() &&
          index < recurrences.size()) {
        failure.template_node_recurrence_route(
            static_cast<std::uint32_t>(index), recurrences[index],
            state->bound.run, step_index);
      }
      return true;
    }
    admitted_steps += step_count;
  }
  return false;
}

PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      const std::span<const PreparedKernelRun *const> runs,
                      const std::span<const std::uint8_t> barriers,
                      const std::span<const std::uint32_t> declared_steps,
                      const std::span<const BackendRecurrence> recurrences,
                      const std::span<const BackendPublish> publications,
                      const std::uint32_t declared_step_count,
                      const std::uint32_t generation_stride,
                      const bool profile_steps,
                      PreparedKernelTemplateRegistry *const templates) {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  PreparedPipelineFailureContext failure{};
  failure.stage(PreparedPipelineFailureStage::CommonValidation);
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      runs.size() != barriers.size() || runs.size() != declared_steps.size() ||
      runs.size() != recurrences.size() || publications.size() > 32u) {
    return reject_pipeline(failure, invalid.reason);
  }
  std::uint32_t state_count = 0u;
  for (const BackendRecurrence &recurrence : recurrences) {
    if (recurrence.window != nullptr) {
      state_count = std::max(state_count, recurrence.window->state + 1u);
    }
  }
  for (const BackendPublish &publication : publications) {
    const auto &target = publication.target;
    const bool window = publication.kind == BackendPublishKind::Window;
    if (publication.target_handle == nullptr ||
        publication.target_ordinal ==
            std::numeric_limits<std::uint32_t>::max() ||
        publication.state >= state_count ||
        (!window && publication.final >= publication.sources.size()) ||
        (window &&
         (publication.maximum == 0u || publication.tile == 0u ||
          publication.tile > publication.maximum ||
          target.count != publication.maximum ||
          publication.count.handle == nullptr ||
          publication.count.source.count != 1u ||
          publication.count.source.element_bytes != sizeof(std::uint32_t) ||
          publication.count.source.stride_bytes < sizeof(std::uint32_t) ||
          publication.count.source.usage !=
              rund::kernel::kResidentUsageRead)) ||
        (window && publication.count_ordinal ==
                       std::numeric_limits<std::uint32_t>::max()) ||
        (!window && (publication.maximum != 0u || publication.tile != 0u ||
                     publication.count_ordinal !=
                         std::numeric_limits<std::uint32_t>::max())) ||
        target.stride_bytes < target.element_bytes ||
        target.usage != rund::kernel::kResidentUsageWrite) {
      return reject_pipeline(failure, invalid.reason);
    }
    for (std::size_t bank = 0u; bank < publication.sources.size(); ++bank) {
      const BackendRead &read = publication.sources[bank];
      const auto &source = read.source;
      if (read.handle == nullptr ||
          publication.source_ordinals[bank] ==
              std::numeric_limits<std::uint32_t>::max() ||
          source.count != (window ? publication.tile : target.count) ||
          source.element_bytes != target.element_bytes ||
          (source.element_bytes != 4u && source.element_bytes != 8u) ||
          source.stride_bytes < source.element_bytes ||
          source.usage != rund::kernel::kResidentUsageRead) {
        return reject_pipeline(failure, invalid.reason);
      }
    }
  }

  failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  if (templates == nullptr || !templates->limit.ok) {
    return reject_pipeline(failure, "accel_kernel_template_invalid");
  }
  PreparedKernelTemplateRegistry &registry = *templates;
  const PreparedKernelPipelineShape runtime_shape =
      runtime_pipeline_shape(publications, recurrences, declared_step_count,
                             generation_stride, profile_steps);
  PreparedKernelPipelineReservation reservation = PlanPreparedKernelPipeline(
      context, runs, recurrences, runtime_shape, &registry);
  if (!reservation.ok) {
    return reject_pipeline(failure, reservation.reason);
  }
  if (reservation.template_step_count > reservation.template_step_capacity) {
    (void)locate_template_step_capacity_failure(
        runs, recurrences, reservation.template_step_capacity, failure);
    return reject_pipeline(failure,
                           PreparedPipelineTemplateStepCapacityReasonKey);
  }
  if (reservation.fingerprint_hi != registry.limit.fingerprint_hi ||
      reservation.fingerprint_lo != registry.limit.fingerprint_lo) {
    return reject_pipeline(failure, "accel_kernel_template_invalid");
  }
  const PreparedKernelPipelineReservation structure =
      plan_pipeline_structure(recurrences);
  PreparedKernelPipelineReservation backend_structure{};
  backend_structure.occurrence_count = structure.occurrence_count;
  backend_structure.window_count = structure.window_count;
  backend_structure.nested_group_count = structure.nested_group_count;
  const rund::AccelCheck backend_planned =
      structure.ok ? plan_runtime_backend_structure(
                         context, runs, recurrences, publications,
                         profile_steps ? declared_step_count : 0u,
                         profile_steps ? structure.occurrence_count : 0u,
                         backend_structure)
                   : rund::AccelCheck{false, structure.reason};
  // The common structure owner already charges these three route dimensions.
  // The backend planner consumes them as shape inputs, then contributes only
  // its retained/native owner fields to the cumulative reservation.
  backend_structure.occurrence_count = 0u;
  backend_structure.window_count = 0u;
  backend_structure.nested_group_count = 0u;
  PreparedKernelPipelineReservation charged_structure = structure;
  if (!structure.ok || !backend_planned.ok ||
      !accumulate_reservation(charged_structure, backend_structure) ||
      !accumulate_reservation(reservation, charged_structure) ||
      !accumulate(reservation.host_bytes, reservation.host_transient_bytes)) {
    return reject_pipeline(failure, !structure.ok
                                        ? structure.reason
                                        : (backend_planned.reason == nullptr
                                               ? "compute_pipeline_capacity"
                                               : backend_planned.reason));
  }
  if (!PreparedKernelPipelineReservationWithin(reservation, registry.limit)) {
    return reject_pipeline(failure, "compute_pipeline_capacity");
  }

  // Freeze every exact shared-template route demand before binding or
  // reserving the registry and before the first backend route can observe a
  // cache miss. Public planning and runtime preparation must have the same
  // unique-route and template partition cardinalities.
  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      template_route_demands{};
  std::uint64_t demand_route_count = 0u;
  std::uint64_t demand_template_count = 0u;
  std::uint64_t complete_route_count = 0u;
  std::uint64_t complete_template_count = 0u;
  if (!plan_backend_template_route_demands(
          runs, generation_stride,
          std::span<BackendTemplateRouteDemand>{template_route_demands.data(),
                                                runs.size()},
          demand_route_count, demand_template_count) ||
      !add(demand_route_count, reservation.map_recurrence.group_count,
           complete_route_count) ||
      !add(demand_template_count, reservation.map_recurrence.template_count,
           complete_template_count) ||
      complete_route_count != reservation.route_count ||
      complete_template_count != reservation.template_count) {
    return reject_pipeline(failure, "accel_kernel_template_invalid");
  }

  const rund::AccelCheck registry_bound =
      BindPreparedKernelTemplateRegistry(context.api, context.id, registry);
  if (!registry_bound.ok) {
    return reject_pipeline(failure, registry_bound.reason);
  }
  PipelineBudgetTransaction budget_transaction{};
  const rund::AccelCheck budget = reserve_pipeline_budget(
      registry, runs, recurrences, charged_structure,
      reservation.map_recurrence, generation_stride, budget_transaction);
  if (!budget.ok) {
    return reject_pipeline(failure, budget.reason);
  }

  std::shared_ptr<prepared::PipelineState> pipeline{};
  try {
    pipeline = std::make_shared<prepared::PipelineState>();
    pipeline->states =
        std::make_unique<std::shared_ptr<prepared::RunState>[]>(runs.size());
  } catch (const std::bad_alloc &) {
    return reject_pipeline(failure, "compute_pipeline_capacity");
  }
  pipeline->context = context;
  pipeline->templates = registry.owner;
  pipeline->size = runs.size();
  std::vector<BackendBatchEntry> batch_templates;
  ExpandedPipeline expanded;
  try {
    batch_templates.resize(runs.size());
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      failure.template_recurrence_route(static_cast<std::uint32_t>(index),
                                        recurrences[index]);
      const PreparedKernelRun *const item = runs[index];
      auto *const state =
          item == nullptr
              ? nullptr
              : static_cast<prepared::RunState *>(item->owner.get());
      const BackendOps *const candidate =
          state == nullptr ? nullptr : state->bound.run.ops;
      if (item == nullptr || !item->ok || state == nullptr ||
          !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
          candidate->prepare_pipeline == nullptr ||
          candidate->submit_prepared_pipeline == nullptr ||
          !prepared::MatchesContext(context, *state) ||
          (pipeline->ops != nullptr && pipeline->ops != candidate)) {
        return reject_pipeline(failure, invalid.reason);
      }
      pipeline->ops = candidate;
      pipeline->states[index] =
          std::static_pointer_cast<prepared::RunState>(item->owner);
      const BackendTemplateRouteDemand demand = template_route_demands[index];
      if (!demand.valid() || state->bound.run.templates != nullptr ||
          state->bound.run.failed_node != nullptr ||
          !state->bound.run.template_route_demand.empty()) {
        return reject_pipeline(failure, "accel_kernel_template_invalid");
      }
      if (state->backend == nullptr) {
        PreparedMemory memory{};
        std::uint32_t failed_node = NoNode;
        const BackendPreparationCursor cursor{state->bound.run, registry,
                                              failed_node, demand};
        const rund::AccelCheck materialized =
            candidate->prepare_pipeline_private(state->bound.run,
                                                state->backend, memory);
        if (!materialized.ok || state->backend == nullptr) {
          PreparedPipelineFailure rejected =
              failure.failure(materialized.reason);
          rejected.node = failed_node;
          return PreparedKernelPipeline{.failure = rejected};
        }
        if (candidate->traffic != nullptr) {
          state->roundtrip.internal_bytes =
              ::rund::detail::counter::SaturatingAdd(
                  state->roundtrip.internal_bytes,
                  candidate->traffic(state->backend));
        }
        state->memory.add(memory);
      }
      batch_templates[index] = BackendBatchEntry{
          .run = &state->bound.run,
          .prepared = &state->backend,
          .recurrence = recurrences[index],
          .template_index = static_cast<std::uint32_t>(index)};
    }
  } catch (const std::bad_alloc &) {
    return reject_pipeline(failure, "compute_pipeline_capacity");
  }
  try {
    if (!expand_pipeline(batch_templates, barriers, publications,
                         declared_steps, declared_step_count, profile_steps,
                         pipeline->ops->nested_aggregate_command_count,
                         expanded)) {
      return reject_pipeline(expanded.failure, expanded.reason);
    }
  } catch (const std::bad_alloc &) {
    return reject_pipeline(expanded.failure, "compute_pipeline_capacity");
  }
  failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  const auto account_current = [&]() -> const char * {
    pipeline->counts = {};
    if (expanded.command_count == 0u ||
        !PreparePipelineStatusLayout(
            pipeline->status, declared_steps, declared_step_count,
            expanded.command_count, generation_stride)) {
      return invalid.reason;
    }
    if (expanded.compact_aggregate) {
      if (expanded.aggregates.size() != 1u) {
        return invalid.reason;
      }
      const NestedAggregate &aggregate = expanded.aggregates.front();
      for (std::size_t index = 0u; index < pipeline->size; ++index) {
        failure.template_route(static_cast<std::uint32_t>(index));
        const std::uint64_t occurrences = aggregate.authored_occurrences(index);
        if (pipeline->states[index] == nullptr || occurrences == 0u) {
          return invalid.reason;
        }
        prepared::Accumulate(pipeline->counts, *pipeline->states[index],
                             occurrences);
      }
      return nullptr;
    }
    for (const BackendBatchEntry &command : expanded.commands) {
      failure.occurrence_route(command);
      if (command.template_index >= pipeline->size ||
          pipeline->states[command.template_index] == nullptr) {
        return invalid.reason;
      }
      if (command.transducer == NoTileTransducer) {
        prepared::Accumulate(pipeline->counts,
                             *pipeline->states[command.template_index]);
        continue;
      }
      if (command.transducer >= expanded.transducers.size()) {
        return invalid.reason;
      }
      const TileTransducer &transducer =
          expanded.transducers[command.transducer];
      const std::uint64_t template_end =
          static_cast<std::uint64_t>(transducer.template_first) +
          transducer.template_count;
      if (transducer.template_count == 0u || template_end > pipeline->size) {
        return invalid.reason;
      }
      // One physical transducer command represents the complete authored
      // Action subrange for this outer window. Evidence remains logical: count
      // every original template once while the backend reports the smaller
      // physical dispatch count independently.
      for (std::uint32_t offset = 0u; offset < transducer.template_count;
           ++offset) {
        const std::size_t template_index = transducer.template_first + offset;
        failure.template_route(static_cast<std::uint32_t>(template_index));
        if (pipeline->states[template_index] == nullptr) {
          return invalid.reason;
        }
        prepared::Accumulate(pipeline->counts,
                             *pipeline->states[template_index]);
      }
    }
    return nullptr;
  };

  if (const char *const reason = account_current(); reason != nullptr) {
    return reject_pipeline(failure, reason);
  }
  PreparedPipelineFailure backend_failure{};
  const rund::AccelCheck built = pipeline->ops->prepare_pipeline(
      batch_templates, expanded.commands, expanded.barriers,
      expanded.transducers, expanded.aggregates, publications, registry,
      pipeline->status, profile_steps, pipeline->backend, pipeline->memory,
      backend_failure);
  if (!built.ok) {
    if (backend_failure.stage == PreparedPipelineFailureStage::Unknown) {
      failure.stage(PreparedPipelineFailureStage::Unknown);
      backend_failure = failure.failure(built.reason);
    }
    return PreparedKernelPipeline{.failure = backend_failure};
  }
  failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  if (!ValidPreparedPipelineStatusLayout(
          pipeline->status, declared_steps, declared_step_count,
          expanded.command_count, generation_stride)) {
    return reject_pipeline(failure, invalid.reason);
  }
  const std::uint64_t common_host_bytes =
      sizeof(prepared::PipelineState) +
      pipeline->size * sizeof(std::shared_ptr<prepared::RunState>);
  accumulate_memory(pipeline->memory.host,
                    PreparedMemory{.current = common_host_bytes,
                                   .peak = common_host_bytes,
                                   .cumulative = common_host_bytes,
                                   .budget = common_host_bytes});
  budget_transaction.commit();
  return PreparedKernelPipeline{
      .owner = std::static_pointer_cast<void>(pipeline),
      .ok = true,
  };
}

rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     const std::uint32_t generation) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->seed_prepared_pipeline_generation == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  return pipeline->ops->seed_prepared_pipeline_generation(pipeline->backend,
                                                          generation);
}

PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->memory
                                            : PreparedPipelineMemory{};
}

PreparedMemory ReadPreparedKernelTemplateRegistryMemory(
    const PreparedKernelTemplateRegistry &registry) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr) {
    return {};
  }
  const std::lock_guard<std::recursive_mutex> lock{state->mutex};
  std::uint64_t entries = 0u;
  std::uint64_t charges = 0u;
  std::uint64_t bytes = sizeof(PreparedKernelTemplateRegistryState);
  if (!multiply(state->entries.capacity(), sizeof(PreparedKernelTemplateEntry),
                entries) ||
      !multiply(state->template_charges.capacity(),
                sizeof(PreparedKernelTemplateCharge), charges) ||
      !accumulate(bytes, entries) || !accumulate(bytes, charges)) {
    bytes = std::numeric_limits<std::uint64_t>::max();
  }
  PreparedMemory memory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
  const auto add_memory = [&](const PreparedMemory value) noexcept {
    memory.current =
        ::rund::detail::counter::SaturatingAdd(memory.current, value.current);
    memory.peak =
        ::rund::detail::counter::SaturatingAdd(memory.peak, value.peak);
    memory.cumulative = ::rund::detail::counter::SaturatingAdd(
        memory.cumulative, value.cumulative);
    memory.reused =
        ::rund::detail::counter::SaturatingAdd(memory.reused, value.reused);
    memory.budget =
        ::rund::detail::counter::SaturatingAdd(memory.budget, value.budget);
  };
  for (std::size_t index = 0u; index < state->entries.size(); ++index) {
    const PreparedKernelTemplateEntry &entry = state->entries[index];
    if (entry.prepared == nullptr || entry.ops == nullptr ||
        entry.ops->observe_pipeline_template == nullptr) {
      add_memory(PreparedMemory{
          .current = std::numeric_limits<std::uint64_t>::max(),
          .peak = std::numeric_limits<std::uint64_t>::max(),
          .cumulative = std::numeric_limits<std::uint64_t>::max(),
          .budget = std::numeric_limits<std::uint64_t>::max(),
      });
      break;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (state->entries[prior].prepared.get() == entry.prepared.get()) {
        first_owner = false;
        break;
      }
    }
    if (first_owner) {
      add_memory(entry.ops->observe_pipeline_template(entry.prepared.get()));
    }
  }
  return memory;
}

PreparedPipelineStatusLayout ReadPreparedKernelPipelineStatus(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->status
                                            : PreparedPipelineStatusLayout{};
}

} // namespace rund::node::accel::detail
