#include "pool.hpp"

#include "../state.hpp"

#include <algorithm>
#include <exception>
#include <new>

namespace rund::compute::detail::residency {

bool Pool::configure_execution() noexcept {
  if (device == nullptr) {
    return false;
  }
  if (device->backend == Backend::Cpu) {
    return executor.configure();
  }
  try {
    accelerator_execution = std::make_unique<AcceleratorExecutionRing>();
    if (!graph_owners.empty() && layout.host_output_frame_capacity != 0u) {
      graph_persist = std::make_unique<GraphPersistRing>();
    }
  } catch (const std::bad_alloc &) {
    return false;
  }
  const bool execution =
      std::all_of(accelerator_execution->banks.begin(),
                  accelerator_execution->banks.end(),
                  [](AcceleratorExecutor &slot) { return slot.configure(); }) &&
      accelerator_execution->service.configure();
  return execution &&
         (graph_persist == nullptr ||
          std::all_of(graph_persist->slots.begin(), graph_persist->slots.end(),
                      [&](Persister &slot) {
                        return slot.configure(layout.output_page_bytes,
                                              layout.frame_capacity);
                      }));
}

bool Pool::submit_execution(const std::uint32_t bank,
                            const std::shared_ptr<PipelineState> &pipeline,
                            const EpochLease lease,
                            const bool defer_generation,
                            const std::uint64_t control_generation,
                            const std::uint8_t control_parity,
                            const std::uint64_t publication_generation,
                            const std::uint8_t publication_parity) noexcept {
  if (device == nullptr || bank >= BankCount) {
    return false;
  }
  return device->backend == Backend::Cpu
             ? executor.submit(pipeline, lease, defer_generation,
                               control_generation, control_parity,
                               publication_generation, publication_parity)
             : accelerator_execution != nullptr &&
                   accelerator_execution->banks[bank].submit(
                       pipeline, lease, defer_generation, control_generation,
                       control_parity, publication_generation,
                       publication_parity);
}

ExecutionReceipt Pool::wait_execution(const std::uint32_t bank) noexcept {
  if (device == nullptr || bank >= BankCount) {
    return ExecutionReceipt{.status = Status::fail(Reason::PipelineInvalid)};
  }
  return device->backend == Backend::Cpu ? executor.wait()
         : accelerator_execution == nullptr
             ? ExecutionReceipt{.status = Status::fail(Reason::PipelineInvalid)}
             : accelerator_execution->banks[bank].wait();
}

bool Pool::submit_recurrent(const AcceleratorServiceTask task,
                            void *const user) noexcept {
  return device != nullptr && device->backend != Backend::Cpu &&
         accelerator_execution != nullptr &&
         accelerator_execution->service.submit(task, user);
}

void Pool::wait_recurrent() noexcept {
  if (accelerator_execution != nullptr) {
    accelerator_execution->service.wait();
  }
}

const PoolPhysicalOwner *
Pool::graph_owner(const std::uint32_t physical_id) const noexcept {
  const auto found = std::lower_bound(
      graph_owners.begin(), graph_owners.end(), physical_id,
      [](const PoolPhysicalOwner &owner, const std::uint32_t id) {
        return owner.physical_id < id;
      });
  return found == graph_owners.end() || found->physical_id != physical_id
             ? nullptr
             : &*found;
}

} // namespace rund::compute::detail::residency
