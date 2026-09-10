#include "state.hpp"

#include "../device/residency/pool.hpp"
#include "../device/residency/registry/cpu_graph_owner.hpp"
#include "../device/residency/registry/graph_persist_owner.hpp"

#include <exception>

namespace rund::compute::detail {

VirtualPipelineState::~VirtualPipelineState() noexcept {
  if (cpu_quarantine_hold == nullptr) {
    return;
  }
  if (pipeline == nullptr || pipeline->residency_pool == nullptr ||
      cpu_receipts == nullptr || cpu_quarantine_hold->book != cpu_receipts) {
    std::terminate();
  }
  residency::Pool &pool = *pipeline->residency_pool;
  if (cpu_quarantine_hold->pipeline != pipeline.get() ||
      cpu_quarantine_hold->pool != &pool ||
      cpu_quarantine_hold->authority != &pool.authority()) {
    std::terminate();
  }
  std::lock_guard lock{pool.execution_gate()};
  auto persists = pool.authority().graph_persists();
  if (cpu_quarantine_hold->empty()) {
    if (!persists.cpu_graph_retry_active()) {
      return;
    }
    if (!persists.discard_cpu_graph_retry(cpu_receipts->domain())) {
      std::terminate();
    }
    return;
  }
  if (!pool.authority().cpu_graph().discard_cpu_quarantine(
          cpu_quarantine_hold, pipeline.get(), &pool)) {
    std::terminate();
  }
}

std::shared_ptr<PipelineState>
graph_stage_pipeline(const VirtualPipelineState &state, const std::size_t stage,
                     const std::size_t bank) noexcept {
  if (bank >= residency::Pool::BankCount ||
      stage >= state.graph_pipelines.size() / residency::Pool::BankCount) {
    return {};
  }
  return state.graph_pipelines[stage * residency::Pool::BankCount + bank];
}

std::shared_ptr<PipelineState>
graph_terminal_pipeline(const VirtualPipelineState &state,
                        const std::size_t bank) noexcept {
  if (bank >= residency::Pool::BankCount ||
      state.graph_pipelines.size() < residency::Pool::BankCount ||
      state.graph_pipelines.size() % residency::Pool::BankCount != 0u) {
    return {};
  }
  return state.graph_pipelines[state.graph_pipelines.size() -
                               residency::Pool::BankCount + bank];
}

bool cpu_graph_ready(VirtualPipelineState &state) noexcept {
  if (state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend != Backend::Cpu ||
      state.pipeline->residency_pool == nullptr ||
      state.cpu_receipts == nullptr || state.cpu_quarantine_hold == nullptr) {
    return false;
  }
  residency::Pool *const pool = state.pipeline->residency_pool.get();
  residency::Authority &authority = pool->authority();
  auto persists = authority.graph_persists();
  const graph_reduce::CpuGraphQuarantine &hold = *state.cpu_quarantine_hold;
  return hold.book == state.cpu_receipts &&
         hold.phase == graph_reduce::CpuGraphQuarantine::Phase::Ready &&
         hold.pipeline == state.pipeline.get() && hold.pool == pool &&
         hold.authority == &authority && hold.empty() &&
         !authority.cpu_graph().cpu_quarantine_active() &&
         !persists.has_foreign_cpu_graph_retry(state.cpu_receipts->domain());
}

} // namespace rund::compute::detail
