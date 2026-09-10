#include "../internal.hpp"

#include "../../../../backing.hpp"
#include "../../../../run/backing.hpp"
#include "../../transfer.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::supply_cpu_stage(
    Ticket &ticket, PipelineState &pipeline, const std::uint32_t stage,
    const residency::EpochLease lease, VirtualSupplyResult &result) noexcept {
  using ::rund::detail::counter::Accumulate;
  result = {};
  if (pipeline.device == nullptr || pipeline.device->backend != Backend::Cpu ||
      pipeline.residency_pool.get() != &pool_ ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      stage >= graph_.stages().size() || ticket.count == 0u ||
      ticket.count > PipelineLeafCapacity || lease.token == 0u ||
      lease.ports.empty()) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result.status;
  }
  const auto &declared_ports = graph_.stages()[stage].ports;
  const auto port_declared =
      [&declared_ports](const residency::GraphLeasePort &port) noexcept {
        return std::find_if(
                   declared_ports.begin(), declared_ports.end(),
                   [port](const residency::TiledGraphPort candidate) noexcept {
                     return candidate.resource == port.resource &&
                            candidate.access == port.access &&
                            candidate.program_port == port.program_port;
                   }) != declared_ports.end();
      };
  bool supplied = false;
  for (const residency::GraphLeasePort &port : lease.ports) {
    if (port.access != residency::Access::Read) {
      continue;
    }
    const residency::TiledGraphResource *const declared =
        graph_.resource(port.resource);
    if (!port_declared(port) || declared == nullptr) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result.status;
    }
    if (declared->kind != residency::GraphResourceKind::ExternalInput ||
        declared->persistence != residency::ResourcePersistence::Backing) {
      continue;
    }
    supplied = true;
    std::size_t input_index = 0u;
    const residency::GraphMaterialization *const materialization =
        graph_materialization(run_, port.resource);
    const residency::PoolPhysicalOwner *const owner =
        pool_.graph_owner(declared->physical_id);
    if (materialization == nullptr || !find_input(port.resource, input_index) ||
        input_index >= input_count_ || inputs_[input_index] == nullptr ||
        owner == nullptr || owner->arena == nullptr ||
        port.cache_region_count != owner->cache_regions.size() ||
        port.cache_regions != owner->cache_regions ||
        port.region != owner->bank_regions[pipeline.residency_bank] ||
        port.region.count != run_.frame_capacity ||
        port.first_binding > lease.bindings.size() ||
        port.binding_count != ticket.count ||
        port.binding_count > lease.bindings.size() - port.first_binding) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result.status;
    }
    VirtualBacking &source = *inputs_[input_index];
    const VirtualRunInputProjection &input = run_.inputs[input_index];
    if (VirtualBackingAccess::id(source) != input.backing ||
        VirtualBackingAccess::version(source) != input.version ||
        source.size_bytes() != input.capacity_bytes ||
        materialization->key.backing != input.backing ||
        materialization->key.version != input.version ||
        materialization->page_bytes != declared->page_bytes ||
        materialization->page_count == 0u) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result.status;
    }
    const residency::FrameRegion bank =
        owner->cache_regions[pipeline.residency_bank];
    if (bank.count == 0u || port.region.first < bank.first ||
        port.region.count > bank.count - (port.region.first - bank.first)) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result.status;
    }
    const std::shared_ptr<BufferState> &storage =
        owner->buffers[pipeline.residency_bank];
    CpuBufferState *const cpu =
        storage == nullptr ? nullptr : cpu_buffer(*storage);
    if (cpu == nullptr || cpu->data == nullptr || cpu->bytes < storage->bytes ||
        materialization->page_bytes > std::numeric_limits<std::size_t>::max()) {
      result.status = Status::fail(Reason::TransferInvalid);
      return result.status;
    }
    for (std::size_t page = 0u; page < port.binding_count; ++page) {
      const residency::CacheBinding &binding =
          lease.bindings[port.first_binding + page];
      residency::CacheKey expected{};
      if (binding.access != residency::Access::Read ||
          !residency::project_graph_cache_key(
              *materialization,
              residency::PageKey{.resource = port.resource,
                                 .page = binding.key.page},
              expected) ||
          binding.key != expected || binding.frame < port.region.first ||
          binding.frame - port.region.first >= port.region.count) {
        result.status = Status::fail(Reason::PipelineInvalid);
        return result.status;
      }
      if (!binding.fetch) {
        continue;
      }
      VirtualInputPageProjection projected{};
      std::uint64_t target_offset = 0u;
      if (!project_virtual_input_page(run_, binding.key.page, projected) ||
          projected.transfer_bytes > materialization->page_bytes ||
          !kernel::checked::mul(
              static_cast<std::uint64_t>(binding.frame - bank.first),
              materialization->page_bytes, target_offset) ||
          !kernel::checked::add(target_offset, projected.target_offset,
                                target_offset) ||
          target_offset > cpu->bytes ||
          materialization->page_bytes > cpu->bytes - target_offset) {
        result.status = Status::fail(Reason::PipelineInvalid);
        return result.status;
      }
      std::byte *const target = cpu->data.get() + target_offset;
      std::memset(target, 0,
                  static_cast<std::size_t>(materialization->page_bytes));
      const Status read =
          source.read(projected.logical_offset,
                      std::span<std::byte>{target + projected.target_offset,
                                           projected.transfer_bytes});
      if (!read) {
        result.status = read;
        return result.status;
      }
      ++result.fetched_pages;
      if (result.backing_bytes > std::numeric_limits<std::uint64_t>::max() -
                                     projected.transfer_bytes) {
        result.status = Status::fail(Reason::PipelineCapacity);
        return result.status;
      }
      result.backing_bytes += projected.transfer_bytes;
      Accumulate(stats_.pipeline.residency.backing_read_bytes,
                 projected.transfer_bytes);
    }
  }
  if (!supplied) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result.status;
  }
  return result.status;
}

} // namespace rund::compute::detail::graph_reduce
