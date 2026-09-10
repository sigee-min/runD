#pragma once

#include "range.hpp"
#include "request.hpp"
#include "service.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool persistent_sliding_service_identity(
    const PersistentResidencySlidingRequest &request,
    const std::uint64_t coordinate,
    PersistentResidencySlidingServiceIdentity &identity) noexcept {
  identity = {};
  if (request.width == 0u ||
      request.width > PersistentResidencySlidingCapacity ||
      coordinate >= request.coordinate_count) {
    return false;
  }
  if (!persistent_sliding_range_valid(
          std::span<const PersistentResidencySlidingRole>{request.roles.data(),
                                                           request.width},
          request.width, request.coordinate_count)) {
    return false;
  }
  const std::uint8_t slot =
      static_cast<std::uint8_t>(coordinate % request.width);
  const std::uint64_t turn = coordinate / request.width;
  const PersistentResidencySlidingRole &role = request.roles[slot];
  if (role.slot != slot) {
    return false;
  }
  identity = PersistentResidencySlidingServiceIdentity{
      .plan_identity = request.plan_identity,
      .token = request.token,
      .generation = request.generation,
      .coordinate = coordinate,
      .turn = turn,
      .descriptor_generation = role.first_descriptor_generation +
                               turn * role.descriptor_generation_stride,
      .control_generation =
          static_cast<std::uint32_t>(role.first_control_generation +
                                     turn * role.control_generation_stride),
      .slot = slot,
  };
  return true;
}

[[nodiscard]] inline bool persistent_sliding_same_service_identity(
    const PersistentResidencySlidingServiceIdentity &left,
    const PersistentResidencySlidingServiceIdentity &right) noexcept {
  return left.plan_identity == right.plan_identity &&
         left.token == right.token && left.generation == right.generation &&
         left.coordinate == right.coordinate && left.turn == right.turn &&
         left.descriptor_generation == right.descriptor_generation &&
         left.control_generation == right.control_generation &&
         left.slot == right.slot;
}

// Caller holds Control::gate. Service operations derive identity from this
// immutable bound role table instead of trusting the submitted Request again.
[[nodiscard]] inline bool persistent_sliding_service_identity(
    const PersistentResidencySlidingControl &control,
    const std::uint64_t coordinate,
    PersistentResidencySlidingServiceIdentity &identity) noexcept {
  identity = {};
  if (!control.active || control.width == 0u ||
      control.width > PersistentResidencySlidingCapacity ||
      coordinate >= control.coordinate_count) {
    return false;
  }
  if (!persistent_sliding_range_valid(
          std::span<const PersistentResidencySlidingRole>{control.roles.data(),
                                                           control.width},
          control.width, control.coordinate_count)) {
    return false;
  }
  const std::uint8_t slot =
      static_cast<std::uint8_t>(coordinate % control.width);
  const std::uint64_t turn = coordinate / control.width;
  const PersistentResidencySlidingRole &role = control.roles[slot];
  if (role.slot != slot) {
    return false;
  }
  identity = PersistentResidencySlidingServiceIdentity{
      .plan_identity = control.plan_identity,
      .token = control.token,
      .generation = control.generation,
      .coordinate = coordinate,
      .turn = turn,
      .descriptor_generation = role.first_descriptor_generation +
                               turn * role.descriptor_generation_stride,
      .control_generation =
          static_cast<std::uint32_t>(role.first_control_generation +
                                     turn * role.control_generation_stride),
      .slot = slot,
  };
  return true;
}

} // namespace rund::node::accel::detail
