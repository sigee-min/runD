#include <rund/compute.hpp>

#include "src/compute/exception.hpp"
#include "src/compute/status.hpp"

#include <array>
#include <cstdint>
#include <new>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {

[[nodiscard]] bool CapacityExceptionClassesAreCanonical() {
  using rund::compute::detail::compute_exception::
      rethrow_unless_capacity_exception;
  const auto accepts = [](const auto &raise) {
    try {
      raise();
    } catch (...) {
      try {
        rethrow_unless_capacity_exception();
        return true;
      } catch (...) {
        return false;
      }
    }
    return false;
  };
  bool unexpected_preserved = false;
  try {
    throw 7;
  } catch (...) {
    try {
      rethrow_unless_capacity_exception();
    } catch (const int value) {
      unexpected_preserved = value == 7;
    } catch (...) {
    }
  }
  return accepts([] { throw std::bad_alloc{}; }) &&
         accepts([] { throw std::length_error{"capacity"}; }) &&
         unexpected_preserved;
}

} // namespace

int RunComputeErrorContract() {
  if (!CapacityExceptionClassesAreCanonical()) {
    return 8;
  }
  const auto success = rund::compute::Status::success();
  const auto binding =
      rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  const auto forged =
      rund::compute::Status::fail(static_cast<rund::compute::Reason>(0xffffu));
  if (!success.error().empty() || binding.error() != "compute_shape_mismatch" ||
      forged.error() != "compute_reason_invalid") {
    return 1;
  }
  const auto device_busy =
      rund::compute::Status::fail(rund::compute::Reason::DeviceBusy);
  if (device_busy.error() != "compute_device_busy") {
    return 6;
  }

  struct Projection final {
    std::string_view diagnostic;
    rund::compute::Reason boundary;
    rund::compute::Reason expected;
  };
  constexpr std::array projections{
      Projection{"accel_buffer_unavailable",
                 rund::compute::Reason::BufferCapacity,
                 rund::compute::Reason::BufferCapacity},
      Projection{"cpu_simd_lowering_invalid",
                 rund::compute::Reason::LoweringInvalid,
                 rund::compute::Reason::LoweringInvalid},
      Projection{"accel_metal_command_unavailable",
                 rund::compute::Reason::BackendFailed,
                 rund::compute::Reason::BackendFailed},
      Projection{"compute_device_busy", rund::compute::Reason::BackendFailed,
                 rund::compute::Reason::DeviceBusy},
      Projection{"compute_reduce_sum_overflow",
                 rund::compute::Reason::TileBackendFailed,
                 rund::compute::Reason::ReduceSumOverflow},
  };
  for (const Projection projection : projections) {
    const rund::compute::Reason projected =
        rund::compute::detail::project_reason(projection.diagnostic,
                                              projection.boundary);
    if (projected != projection.expected ||
        projected == rund::compute::Reason::ReasonInvalid ||
        (projection.diagnostic == "accel_metal_command_unavailable" &&
         (projected == rund::compute::Reason::RuntimeBusy ||
          projected == rund::compute::Reason::DeviceBusy))) {
      return 7;
    }
  }

  auto first = rund::compute::open(rund::compute::Target::cpu());
  auto second = rund::compute::open(rund::compute::Target::cpu());
  if (!first || !second) {
    return 2;
  }

  auto unnamed =
      rund::compute::on(first.value())
          .map<std::int32_t>("", 4, [](auto value) { return value + 1; })
          .compile();
  if (unnamed || unnamed.code() != rund::compute::Code::Invalid ||
      unnamed.error() != "compute_name_empty") {
    return 3;
  }

  auto map = rund::compute::on(first.value())
                 .map<std::int32_t>("identity", 4,
                                    [](auto value) { return value + 0; })
                 .compile();
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto source = second.value().upload(std::span<const std::int32_t>{input});
  auto target = first.value().buffer<std::int32_t>(input.size());
  if (!map || !source || !target) {
    return 4;
  }
  auto mixed = map.value().run(source.value(), target.value());
  if (mixed || mixed.code() != rund::compute::Code::Binding ||
      mixed.error() != "compute_binding_device_mismatch") {
    return 5;
  }

  return 0;
}
