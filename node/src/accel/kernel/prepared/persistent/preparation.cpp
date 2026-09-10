#include "../interface/api.hpp"

#include "../model.hpp"

#include <atomic>
#include <limits>
#include <new>

namespace {

constexpr std::uint64_t PersistentSlidingCellDomain = 0x52554e4450534c31ull;
std::atomic<std::uint64_t> NextPersistentSlidingCell{1u};

[[nodiscard]] std::uint64_t mint_cell() noexcept {
  std::uint64_t current =
      NextPersistentSlidingCell.load(std::memory_order_relaxed);
  for (;;) {
    if (current == 0u || current == std::numeric_limits<std::uint64_t>::max()) {
      return 0u;
    }
    if (NextPersistentSlidingCell.compare_exchange_weak(
            current, current + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return current;
    }
  }
}

} // namespace

namespace rund::node::accel::detail {

PreparedResidencyPersistentSlidingPreparation
PrepareKernelPipelinePersistentSliding(
    const PersistentResidencySlidingRequest &prototype,
    const std::span<const PreparedResidencyPersistentSlidingRole>
        roles) noexcept {
  PreparedResidencyPersistentSlidingPreparation result{};
  if (prototype.width != roles.size() || prototype.lowering != nullptr) {
    result.backend.capability.memory = prototype.memory;
    result.backend.capability.mode = prototype.mode;
    result.backend.capability.check = {false, "accel_kernel_pipeline_invalid"};
    return result;
  }
  result.backend.capability =
      QueryPreparedKernelPipelinePersistentSlidingCapability(
          roles, prototype.coordinate_count, prototype.memory, prototype.mode);
  if (!result.backend.capability.check.ok) {
    return result;
  }
  if (!persistent_sliding_product_capable(result.backend.capability)) {
    result.backend.capability.check = {false, "accel_kernel_pipeline_invalid"};
    return result;
  }
  try {
    result.backend.cell =
        std::make_shared<PersistentResidencySlidingRequestCell>();
    result.backend.ticket =
        std::make_shared<PersistentResidencySlidingTicket>();
  } catch (const std::bad_alloc &) {
    result.backend.capability.check = {false, "compute_pipeline_capacity"};
    return result;
  }
  result.backend.cell->id = mint_cell();
  result.backend.cell->domain = PersistentSlidingCellDomain;
  if (result.backend.cell->id == 0u) {
    result.backend.capability.check = {false, "compute_pipeline_capacity"};
    return result;
  }
  result.backend.ticket->cell = result.backend.cell;
  result.backend.cell->committed = prototype;
  result.backend.cell->committed.cell_id = result.backend.cell->id;
  result.backend.cell->committed.cell_domain = result.backend.cell->domain;
  result.backend.cell->committed.ticket.reset();
  PersistentResidencySlidingRequest request = prototype;
  request.cell_id = result.backend.cell->id;
  request.cell_domain = result.backend.cell->domain;
  request.ticket = result.backend.ticket;
  const auto *const first = static_cast<const prepared::PipelineState *>(
      roles[0u].pipeline.owner.get());
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PreparedResidencyPersistentSlidingRole &source = roles[slot];
    const auto *const state = static_cast<const prepared::PipelineState *>(
        source.pipeline.owner.get());
    request.roles[slot] = PersistentResidencySlidingRole{
        .prepared = state->backend,
        .locals = source.locals,
        .local_count = source.local_count,
        .first_control_generation = source.first_control_generation,
        .control_generation_stride = source.control_generation_stride,
        .first_descriptor_generation = source.first_descriptor_generation,
        .descriptor_generation_stride = source.descriptor_generation_stride,
        .slot = source.slot,
    };
  }
  result.backend.cell->committed.roles = request.roles;
  result.backend.cell->committed.lowering = nullptr;
  const auto cell = result.backend.cell;
  const auto ticket = result.backend.ticket;
  auto native = first->ops->prepare_persistent_sliding(request);
  result.backend = std::move(native);
  result.backend.cell = cell;
  result.backend.ticket = ticket;
  result.backend.cell->committed.lowering.reset();
  if (!result) {
    // Keep the backend's typed preparation reason across the common wrapper.
    // In particular, Vulkan/Metal use compute_pipeline_capacity for a cold
    // stream-capacity rejection; erasing it here would turn a terminal
    // capacity failure into a generic BackendUnsupported at the compute seam.
    const rund::AccelCheck failure = result.backend.capability.check;
    result = {};
    result.backend.capability.check = failure;
  }
  return result;
}

} // namespace rund::node::accel::detail
