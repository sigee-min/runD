#include "local.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../state.hpp"

#include <limits>
#include <memory>
#include <span>

namespace rund::compute::detail {

Status build_pipeline_execution_schedule(
    const residency::execution::Plan &plan,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    PipelineExecutionSchedulePrepared &prepared,
    const bool lower_backend) noexcept {
  constexpr std::size_t Roles =
      node::accel::detail::ResidencyScheduleRoleCapacity;
  prepared = {};
  if (plan.identity() == 0u || plan.epoch_count() <= Roles) {
    return Status::fail(Reason::BackendUnsupported);
  }
  PipelineExecutionSnapshot snapshot{};
  const Status snapshotted = snapshot_pipeline_execution(pipelines, snapshot);
  if (!snapshotted) {
    return snapshotted;
  }
  const std::shared_ptr<DeviceState> &device = pipelines[0u]->device;
  if (device == nullptr || device->ops == nullptr ||
      device->ops->residency.prepare_residency_schedule == nullptr ||
      device->ops->residency.submit_residency_schedule == nullptr ||
      device->ops->residency.signal_residency_schedule == nullptr ||
      device->ops->residency.abort_residency_schedule == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  prepared.request.plan_identity = plan.identity();
  prepared.request.epoch_count = plan.epoch_count();
  prepared.request.role_count = Roles;
  for (std::size_t role = 0u; role < Roles; ++role) {
    residency::execution::Node dispatch{};
    if (!plan.project(
            residency::execution::NodeId{
                .epoch = role, .phase = residency::execution::Phase::Dispatch},
            dispatch) ||
        dispatch.domain != residency::execution::Domain::Native ||
        dispatch.input_count == 0u ||
        dispatch.input_count != dispatch.output_count ||
        dispatch.input_count > residency::execution::UseCapacity) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bank = dispatch.bank;
    if (bank >= pipelines.size() ||
        dispatch.input_count > pipelines[bank]->steps.size() ||
        snapshot.generation[bank] >=
            std::numeric_limits<std::uint32_t>::max()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const node::accel::detail::PreparedKernelPipeline *const native =
        prepared_pipeline_for(*pipelines[bank], snapshot.parity[bank]);
    if (native == nullptr || !native->ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    node::accel::detail::PreparedResidencyScheduleRole &target =
        prepared.request.roles[role];
    target.pipeline = *native;
    target.local_count = dispatch.input_count;
    target.first_control_generation =
        static_cast<std::uint32_t>(snapshot.generation[bank] + 1u);
    target.control_generation_stride = 2u;
    target.role = static_cast<std::uint8_t>(role);
    target.bank = static_cast<std::uint8_t>(bank);
    for (std::size_t local = 0u; local < dispatch.input_count; ++local) {
      target.locals[local] = static_cast<std::uint32_t>(local);
    }
    prepared.pipelines[role] = pipelines[bank];
    ++snapshot.generation[bank];
    if (pipelines[bank]->transactional) {
      snapshot.parity[bank] ^= 1u;
    }
  }
  residency::execution::Node tail{};
  const std::size_t tail_role = static_cast<std::size_t>(
      (plan.epoch_count() - 1u) % static_cast<std::uint64_t>(Roles));
  if (!plan.project(
          residency::execution::NodeId{
              .epoch = plan.epoch_count() - 1u,
              .phase = residency::execution::Phase::Dispatch},
          tail) ||
      tail.input_count == 0u ||
      tail.input_count > prepared.request.roles[tail_role].local_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  prepared.request.tail_local_count = tail.input_count;
  for (std::size_t role = 0u; role < Roles; ++role) {
    const std::uint64_t last = plan.epoch_count() - 1u;
    const std::uint64_t last_role_epoch =
        last - ((last - role) % static_cast<std::uint64_t>(Roles));
    std::uint32_t ignored = 0u;
    if (!pipeline_schedule_generation_for(prepared.request.roles[role],
                                          last_role_epoch, ignored)) {
      return Status::fail(Reason::PipelineCapacity);
    }
  }
  if (lower_backend) {
    // Freeze the whole remaining budget before a backend materializes a cold
    // lowering. Once its exact retained charge is known, partition that exact
    // charge and refund the unused capacity. No Authority token exists in this
    // phase and no concurrent preparation can oversubscribe the same budget.
    const storage::Report report = device->pipeline_memory_budget.report();
    if (!report) {
      prepared = {};
      return Status::fail(Reason::DevicePipelineMemoryCapacity);
    }
    storage::Reservation capacity =
        device->pipeline_memory_budget.reserve(report.available_bytes);
    if (!capacity) {
      prepared = {};
      return Status::fail(Reason::DevicePipelineMemoryCapacity);
    }
    prepared.lowering = device->ops->residency.prepare_residency_schedule(
        *device, prepared.request.roles, prepared.request.epoch_count,
        prepared.request.tail_local_count);
    const bool storage_overflow = prepared.lowering.retained_bytes >
                                  std::numeric_limits<std::uint64_t>::max() -
                                      prepared.lowering.transient_bytes;
    const std::uint64_t admitted_bytes =
        storage_overflow ? std::numeric_limits<std::uint64_t>::max()
                         : prepared.lowering.retained_bytes +
                               prepared.lowering.transient_bytes;
    if (!prepared.lowering.check.ok || !prepared.lowering.callbacks_async ||
        prepared.lowering.kind ==
            node::accel::detail::BackendResidencyScheduleLowering::
                Unsupported ||
        prepared.lowering.queue_calls == 0u || storage_overflow ||
        admitted_bytes > capacity.max_allocated_bytes()) {
      const bool over_budget =
          storage_overflow || admitted_bytes > capacity.max_allocated_bytes();
      const Reason reason = project_reason(prepared.lowering.check.reason,
                                           Reason::BackendUnsupported);
      prepared = {};
      return Status::fail(over_budget ? Reason::DevicePipelineMemoryCapacity
                                      : reason);
    }
    // Transient submit-time record payload is still real live Host storage.
    // Reserve the backend-reported logical payload bound with retained native
    // storage before Authority begins; a huge-Q Vulkan candidate must fall
    // back before any execution token rather than allocate Q arrays after
    // admission. Allocator metadata/capacity rounding is backend overhead, so
    // this field is not labelled as an exact process-heap footprint.
    storage::Reservation exact = capacity.partition(admitted_bytes);
    if (!exact || !capacity.refund()) {
      prepared = {};
      return Status::fail(Reason::PipelineInvalid);
    }
    const storage::Status committed = exact.commit(storage::Usage{
        .physical_bytes = 0u,
        .allocated_bytes = admitted_bytes,
    });
    if (!committed) {
      prepared = {};
      return Status::fail(Reason::PipelineInvalid);
    }
    try {
      prepared.memory =
          std::make_shared<storage::Reservation>(std::move(exact));
    } catch (const std::bad_alloc &) {
      prepared = {};
      return Status::fail(Reason::PipelineMemoryBudget);
    }
    prepared.request.lowering = prepared.lowering.owner;
    prepared.request.admission = prepared.memory;
  }
  return Status::success();
}

Status prepare_pipeline_execution_schedule_impl(
    const residency::execution::Plan &plan,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    PipelineExecutionSchedulePrepared &prepared) noexcept {
  return build_pipeline_execution_schedule(plan, pipelines, prepared, true);
}

} // namespace rund::compute::detail
