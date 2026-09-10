#include "local.hpp"

#include "src/accel/backend/token.hpp"
#include "src/accel/cpu/buffer.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <memory>
#include <mutex>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

} // namespace

bool CheckCpuCounters(const rund::AccelDevice &pick) {
  const std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  detail::CpuAdapter *const adapter =
      raw == nullptr ? nullptr : detail::CpuAdapterFromPick(*raw);
  if (adapter == nullptr) {
    return false;
  }
  {
    std::lock_guard lock{adapter->mutex};
    OverflowCounter(adapter->dispatch_count);
    OverflowCounter(adapter->buffer_allocation_count);
    OverflowCounter(adapter->host_to_device_bytes);
    OverflowCounter(adapter->device_to_host_bytes);
  }
  const rund::RuntimeStats saturated =
      rund::node::accel::ReadRuntimeStats(pick);
  const bool saturated_ok =
      saturated.outcome.ok &&
      CountersEqual(kCounterMaximum,
                    {saturated.run.work.dispatch_count,
                     saturated.run.allocations.buffer_allocation_count,
                     saturated.run.transfer.host_to_device_bytes,
                     saturated.run.transfer.device_to_host_bytes});
  rund::node::accel::ResetRuntimeStats(pick);
  const rund::RuntimeStats reset = rund::node::accel::ReadRuntimeStats(pick);
  return saturated_ok && reset.outcome.ok &&
         CountersEqual(0u, {reset.run.work.dispatch_count,
                            reset.run.allocations.buffer_allocation_count,
                            reset.run.transfer.host_to_device_bytes,
                            reset.run.transfer.device_to_host_bytes});
}

} // namespace node_accel_contract::backend_runtime
