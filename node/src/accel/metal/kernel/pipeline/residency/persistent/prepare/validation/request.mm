#include "../../internal.hpp"

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

std::uint64_t first_invalid_request_structure(
    const PersistentResidencySlidingRequest &request,
    MetalAdapter *&adapter) noexcept {
  std::array<NativeRole, PersistentResidencySlidingCapacity> roles{};
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PersistentResidencySlidingRole &source = request.roles[slot];
    NativeRole &role = roles[slot];
    role.sequence = static_cast<const MetalSequence *>(source.prepared.get());
    role.locals = source.locals;
    role.local_count = source.local_count;
    role.first_control_generation = source.first_control_generation;
    role.control_generation_stride = source.control_generation_stride;
    role.first_descriptor_generation = source.first_descriptor_generation;
    role.descriptor_generation_stride = source.descriptor_generation_stride;
    role.slot = source.slot;
    role.pipeline_ok = source.prepared != nullptr;
  }
  return first_invalid_structure(std::span<const NativeRole>{roles},
                                 request.width, request.coordinate_count,
                                 request.memory, request.mode, adapter);
}

} // namespace

std::uint64_t
first_invalid_request(const PersistentResidencySlidingRequest &request,
                      MetalAdapter *&adapter) noexcept {
  adapter = nullptr;
  if (request.plan_identity == 0u)
    return request_issue_key(RequestIssue::PlanIdentity);
  if (request.token == 0u)
    return request_issue_key(RequestIssue::Token);
  if (request.generation == 0u)
    return request_issue_key(RequestIssue::Generation);
  if (request.owner_nonce == 0u)
    return request_issue_key(RequestIssue::OwnerNonce);
  if (request.coordinate_count == 0u)
    return request_issue_key(RequestIssue::CoordinateCount);
  if (request.tail_local_count == 0u)
    return request_issue_key(RequestIssue::TailLocalCount);
  if (request.admission == nullptr)
    return request_issue_key(RequestIssue::Admission);
  if (request.final == nullptr)
    return request_issue_key(RequestIssue::Final);
  if (request.user == nullptr)
    return request_issue_key(RequestIssue::User);
  if (request.memory != ResidencySlidingMemory::HostCoherent)
    return request_issue_key(RequestIssue::Memory);
  if ((request.mode != PersistentResidencySlidingMode::OneSubmit &&
       request.mode != PersistentResidencySlidingMode::BackendChunked) ||
      (request.mode == PersistentResidencySlidingMode::OneSubmit &&
       (request.chunk_count != 0u || request.first_coordinate != 0u)) ||
      (request.mode == PersistentResidencySlidingMode::BackendChunked &&
       (request.chunk_count == 0u || request.chunk_count > 2u ||
        request.chunk_count > request.coordinate_count ||
        request.first_coordinate >= request.coordinate_count ||
        request.first_coordinate >
            request.coordinate_count - request.chunk_count)))
    return request_issue_key(RequestIssue::Mode);
  const std::uint64_t structure =
      first_invalid_request_structure(request, adapter);
  if (structure != request_issue_key(RequestIssue::Valid))
    return structure;
  const std::size_t tail_slot =
      static_cast<std::size_t>((request.coordinate_count - 1u) % request.width);
  if (request.tail_local_count > request.roles[tail_slot].local_count)
    return request_issue_key(RequestIssue::TailCapacity);
  return request_issue_key(RequestIssue::Valid);
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
