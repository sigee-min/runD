#include "local.hpp"

#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckValidationAndIdentity(rund::compute::Device &device) {
  using namespace rund::compute;
  using Real = Fixed<20, 44>;
  constexpr std::array<std::int32_t, 4u> values{1, 2, 3, 4};
  auto first = on(device)
                   .map<std::int32_t>("pipeline-identity-first", values.size(),
                                      [](auto value) { return value * 2; })
                   .compile();
  auto second =
      on(device)
          .map<std::int32_t>("pipeline-identity-second", values.size(),
                             [](auto value) { return value + 3; })
          .compile();
  auto alternate =
      on(device)
          .map<std::int32_t>("pipeline-identity-alternate", values.size(),
                             [](auto value) { return value + 4; })
          .compile();
  auto input = Upload(device, values);
  auto middle = device.buffer<std::int32_t>(values.size());
  auto output = device.buffer<std::int32_t>(values.size());
  if (!first || !second || !alternate || !input || !middle || !output) {
    return 1;
  }
  auto canonical = pipeline(device)
                       .then(*first, read(*input), write(*middle))
                       .then(*second, read(*middle), write(*output))
                       .prepare();
  if (!canonical) {
    return 2;
  }

  auto other_output = device.buffer<std::int32_t>(values.size());
  auto other_input = Upload(device, values);
  auto other_middle = device.buffer<std::int32_t>(values.size());
  if (!other_output || !other_input || !other_middle) {
    return 3;
  }
  auto equivalent =
      pipeline(device)
          .then(*first, read(*other_input), write(*other_middle))
          .then(*second, read(*other_middle), write(*other_output))
          .prepare();
  if (!equivalent || equivalent->fingerprint() != canonical->fingerprint()) {
    return 4;
  }
  auto order_changed =
      pipeline(device)
          .then(*second, read(*other_input), write(*other_middle))
          .then(*first, read(*other_middle), write(*other_output))
          .prepare();
  auto program_changed =
      pipeline(device)
          .then(*first, read(*other_input), write(*other_middle))
          .then(*alternate, read(*other_middle), write(*other_output))
          .prepare();
  auto topology_changed =
      pipeline(device)
          .then(*first, read(*other_input), write(*other_middle))
          .then(*second, read(*other_middle), write(*other_input))
          .prepare();
  if (!order_changed || !program_changed || !topology_changed ||
      order_changed->fingerprint() == canonical->fingerprint() ||
      program_changed->fingerprint() == canonical->fingerprint() ||
      topology_changed->fingerprint() == canonical->fingerprint()) {
    return 5;
  }

  auto alias =
      pipeline(device).then(*first, read(*input), write(*input)).prepare();
  if (alias || alias.reason() != Reason::BindingAliasUnsupported) {
    return 6;
  }
  auto wrong_shape = device.buffer<std::int32_t>(values.size() + 1u);
  if (!wrong_shape) {
    return 7;
  }
  auto shape = pipeline(device)
                   .then(*first, read(*wrong_shape), write(*middle))
                   .prepare();
  if (shape || shape.reason() != Reason::ShapeMismatch) {
    return 8;
  }
  auto other_device = open(Target::cpu(2u));
  auto foreign = other_device ? other_device->upload<std::int32_t>(values)
                              : decltype(other_device->upload<std::int32_t>(
                                    values))::fail(other_device.reason());
  if (!foreign) {
    return 9;
  }
  auto device_mismatch =
      pipeline(device).then(*first, read(*foreign), write(*middle)).prepare();
  if (device_mismatch ||
      device_mismatch.reason() != Reason::BindingDeviceMismatch) {
    return 10;
  }

  auto empty = pipeline(device).prepare();
  if (empty || empty.reason() != Reason::PipelineEmpty) {
    return 11;
  }
  auto capacity_builder = pipeline(device);
  for (std::size_t index = 0u; index < 65u; ++index) {
    if ((index & 1u) == 0u) {
      capacity_builder.then(*first, read(*input), write(*middle));
    } else {
      capacity_builder.then(*first, read(*middle), write(*input));
    }
  }
  auto capacity = std::move(capacity_builder).prepare();
  if (capacity || capacity.reason() != Reason::PipelineCapacity) {
    return 12;
  }

  auto movable =
      pipeline(device).then(*first, read(*input), write(*middle)).prepare();
  if (!movable) {
    return 13;
  }
  Pipeline moved = std::move(*movable);
  if (movable->run() || movable->run().reason() != Reason::PipelineInvalid ||
      !moved.valid()) {
    return 14;
  }

  constexpr std::array<Real, 1u> fixed_values{Real::from_raw(3)};
  auto fixed_input = Upload(device, fixed_values);
  auto fixed_middle = device.buffer<Real>(fixed_values.size());
  auto fixed_output = device.buffer<Real>(fixed_values.size());
  auto policy_consumer =
      on(device)
          .map<Real>("pipeline-policy-consumer", fixed_values.size(),
                     [](auto value) { return quantize<Real>(value); })
          .compile();
  if (!fixed_input || !fixed_middle || !fixed_output || !policy_consumer) {
    return 15;
  }
  const auto policy_rejected = [&]<Rounding Round, Overflow OverflowMode,
                                   Approximation ApproximationMode>() {
    auto program =
        on(device)
            .map<Real>(
                "pipeline-policy", fixed_values.size(),
                [](auto value) {
                  return quantize<Real, Round, OverflowMode, ApproximationMode>(
                      value);
                })
            .compile();
    if (!program) {
      return false;
    }
    auto prepared =
        pipeline(device)
            .then(*program, read(*fixed_input), write(*fixed_middle))
            .then(*policy_consumer, read(*fixed_middle), write(*fixed_output))
            .prepare();
    return !prepared && prepared.reason() == Reason::FixedFormatMismatch;
  };
  if (!policy_rejected.template
       operator()<Rounding::Down, Overflow::Saturate, Approximation::Exact>() ||
      !policy_rejected.template operator()<
          Rounding::NearestEven, Overflow::Wrap, Approximation::Exact>() ||
      !policy_rejected
           .template operator()<Rounding::NearestEven, Overflow::Saturate,
                                Approximation::Deterministic>()) {
    return 16;
  }

  auto pending_read_before_overwrite =
      pipeline(device)
          .state(*input, *middle)
          .then(*first, read(*middle), write(*output))
          .then(*first, read(*input), write(*middle))
          .commit()
          .prepare();
  if (pending_read_before_overwrite ||
      pending_read_before_overwrite.reason() != Reason::PipelineInvalid) {
    return 17;
  }
  auto published_write = pipeline(device)
                             .state(*input, *middle)
                             .then(*first, read(*output), write(*input))
                             .then(*first, read(*input), write(*middle))
                             .commit()
                             .prepare();
  if (published_write || published_write.reason() != Reason::PipelineInvalid) {
    return 18;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
