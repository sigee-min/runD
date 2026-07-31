#include "model.hpp"

#include <rund/compute/math.hpp>

#include <array>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] int CheckPolicy(rund::compute::Device *const device) {
  using FixedValue = rund::compute::Fixed<16, 16>;
  using AlternateFormat = rund::compute::Fixed<17, 15>;
  using SecondAlternateFormat = rund::compute::Fixed<18, 14>;
  auto policy_cache = rund::compute::program_cache(*device, 6u);
  if (!policy_cache) {
    return 37;
  }
  auto exact_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>(
              "fixed-policy-exact", 1u,
              [](auto value) {
                return rund::compute::quantize<
                    FixedValue, rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Exact>(value);
              })
          .compile();
  auto deterministic_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>(
              "fixed-policy-deterministic", 1u,
              [](auto value) {
                return rund::compute::quantize<
                    FixedValue, rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Deterministic>(value);
              })
          .compile();
  auto format_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>(
              "fixed-policy-format", 1u,
              [](auto value) {
                return rund::compute::quantize<
                    AlternateFormat, rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Exact>(value);
              })
          .compile();
  auto rounding_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>("fixed-policy-rounding", 1u,
                           [](auto value) {
                             return rund::compute::quantize<
                                 FixedValue, rund::compute::Rounding::Down,
                                 rund::compute::Overflow::Saturate,
                                 rund::compute::Approximation::Exact>(value);
                           })
          .compile();
  auto overflow_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>(
              "fixed-policy-overflow", 1u,
              [](auto value) {
                return rund::compute::quantize<
                    FixedValue, rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Wrap,
                    rund::compute::Approximation::Exact>(value);
              })
          .compile();
  auto nonlinear_policy =
      rund::compute::on(*device, *policy_cache)
          .map<FixedValue>(
              "fixed-policy-nonlinear", 1u,
              [](auto value) {
                return rund::compute::quantize<
                    FixedValue, rund::compute::Rounding::NearestEven,
                    rund::compute::Overflow::Saturate,
                    rund::compute::Approximation::Deterministic>(
                    rund::compute::sqrt(value));
              })
          .compile();
  if (!exact_policy || !format_policy || !rounding_policy || !overflow_policy ||
      !deterministic_policy || !nonlinear_policy ||
      exact_policy->fingerprint() == format_policy->fingerprint() ||
      exact_policy->fingerprint() == rounding_policy->fingerprint() ||
      exact_policy->fingerprint() == overflow_policy->fingerprint() ||
      exact_policy->fingerprint() == deterministic_policy->fingerprint() ||
      deterministic_policy->fingerprint() == nonlinear_policy->fingerprint() ||
      policy_cache->stats().misses != 6u) {
    return 38;
  }
  const auto format_graph = format_policy->graph();
  const auto rounding_graph = rounding_policy->graph();
  const auto overflow_graph = overflow_policy->graph();
  const auto deterministic_graph = deterministic_policy->graph();
  const auto nonlinear_graph = nonlinear_policy->graph();
  if (format_graph.outputs.size() != 1u ||
      rounding_graph.outputs.size() != 1u ||
      overflow_graph.outputs.size() != 1u ||
      deterministic_graph.outputs.size() != 1u ||
      nonlinear_graph.outputs.size() != 1u) {
    return 39;
  }
  const auto &format_output =
      format_graph.resources[format_graph.outputs.front() - 1u];
  const auto &rounding_output =
      rounding_graph.resources[rounding_graph.outputs.front() - 1u];
  const auto &overflow_output =
      overflow_graph.resources[overflow_graph.outputs.front() - 1u];
  const auto &deterministic_output =
      deterministic_graph.resources[deterministic_graph.outputs.front() - 1u];
  const auto &nonlinear_output =
      nonlinear_graph.resources[nonlinear_graph.outputs.front() - 1u];
  if (format_output.integer_bits != 17u || format_output.fraction_bits != 15u ||
      rounding_output.rounding != rund::compute::Rounding::Down ||
      overflow_output.overflow != rund::compute::Overflow::Wrap ||
      deterministic_output.integer_bits != 16u ||
      deterministic_output.fraction_bits != 16u ||
      deterministic_output.approximation !=
          rund::compute::Approximation::Deterministic ||
      nonlinear_output.approximation !=
          rund::compute::Approximation::Deterministic) {
    return 40;
  }

  auto fixed_identity = rund::compute::on(*device)
                            .map<FixedValue>("fixed-identity-quantize", 4u,
                                             [](auto value) { return value; })
                            .compile();
  const std::array<FixedValue, 4u> fixed_identity_input{
      FixedValue::min(), FixedValue::from_raw(-1), FixedValue::from_raw(1),
      FixedValue::max()};
  if (!fixed_identity) {
    return 58;
  }
  auto fixed_identity_output = fixed_identity->run(fixed_identity_input);
  if (!fixed_identity_output ||
      *fixed_identity_output !=
          std::vector<FixedValue>{fixed_identity_input.begin(),
                                  fixed_identity_input.end()}) {
    return 58;
  }

  auto fixed_constant_cache = rund::compute::program_cache(*device, 2u);
  if (!fixed_constant_cache) {
    return 53;
  }
  const AlternateFormat alternate_one = AlternateFormat::from_raw(1 << 15);
  auto fixed_constant_program =
      rund::compute::on(*device, *fixed_constant_cache)
          .input<FixedValue>(4u)
          .zip_input<AlternateFormat>(4u)
          .branch([alternate_one](auto left, auto right) {
            return rund::compute::zip(left, right)
                .map("fixed-constant-format",
                     rund::compute::capture(
                         [](auto first, auto second, auto constant) {
                           const auto converted =
                               rund::compute::quantize<AlternateFormat>(first);
                           return rund::compute::quantize<
                               FixedValue, rund::compute::Rounding::Down,
                               rund::compute::Overflow::Wrap,
                               rund::compute::Approximation::Exact>(
                               converted - converted + second + constant);
                         },
                         alternate_one));
          })
          .compile();
  if (!fixed_constant_program) {
    return 54;
  }
  const std::array<FixedValue, 4u> fixed_constant_left{};
  const std::array<AlternateFormat, 4u> fixed_constant_right{
      alternate_one, alternate_one, alternate_one, alternate_one};
  auto fixed_constant_job = fixed_constant_program->resident(
      fixed_constant_left, fixed_constant_right);
  if (!fixed_constant_job || !fixed_constant_job->run()) {
    return 55;
  }
  auto fixed_constant_output = fixed_constant_job->read();
  const FixedValue fixed_two = FixedValue::from_raw(2 << 16);
  if (!fixed_constant_output ||
      *fixed_constant_output !=
          std::vector<FixedValue>{fixed_two, fixed_two, fixed_two, fixed_two}) {
    return 56;
  }
  const Info fixed_constant_graph = fixed_constant_program->graph();
  const auto &fixed_constant_resource =
      fixed_constant_graph.resources[fixed_constant_graph.outputs.front() - 1u];
  if (fixed_constant_resource.integer_bits != 16u ||
      fixed_constant_resource.fraction_bits != 16u ||
      fixed_constant_resource.rounding != rund::compute::Rounding::Down ||
      fixed_constant_resource.overflow != rund::compute::Overflow::Wrap) {
    return 57;
  }

  auto second_input_format_cache = rund::compute::program_cache(*device, 2u);
  if (!second_input_format_cache) {
    return 66;
  }
  auto alternate_second_program =
      rund::compute::on(*device, *second_input_format_cache)
          .input<FixedValue>(4u)
          .zip_input<AlternateFormat>(4u)
          .branch([](auto left, auto right) {
            return rund::compute::zip(left, right)
                .map("fixed-second-format-a", [](auto first, auto second) {
                  const auto zero =
                      rund::compute::quantize<FixedValue,
                                              rund::compute::Rounding::Down,
                                              rund::compute::Overflow::Wrap>(
                          first - first);
                  const auto converted =
                      rund::compute::quantize<FixedValue,
                                              rund::compute::Rounding::Down,
                                              rund::compute::Overflow::Wrap>(
                          second);
                  return rund::compute::quantize<FixedValue,
                                                 rund::compute::Rounding::Down,
                                                 rund::compute::Overflow::Wrap>(
                      zero + converted);
                });
          })
          .compile();
  auto other_second_program =
      rund::compute::on(*device, *second_input_format_cache)
          .input<FixedValue>(4u)
          .zip_input<SecondAlternateFormat>(4u)
          .branch([](auto left, auto right) {
            return rund::compute::zip(left, right)
                .map("fixed-second-format-b", [](auto first, auto second) {
                  const auto zero =
                      rund::compute::quantize<FixedValue,
                                              rund::compute::Rounding::Down,
                                              rund::compute::Overflow::Wrap>(
                          first - first);
                  const auto converted =
                      rund::compute::quantize<FixedValue,
                                              rund::compute::Rounding::Down,
                                              rund::compute::Overflow::Wrap>(
                          second);
                  return rund::compute::quantize<FixedValue,
                                                 rund::compute::Rounding::Down,
                                                 rund::compute::Overflow::Wrap>(
                      zero + converted);
                });
          })
          .compile();
  const auto second_input_stats = second_input_format_cache->stats();
  if (!alternate_second_program || !other_second_program ||
      alternate_second_program->fingerprint() ==
          other_second_program->fingerprint() ||
      second_input_stats.misses != 2u || second_input_stats.hits != 0u) {
    return 67;
  }
  const Info alternate_second_graph = alternate_second_program->graph();
  const Info other_second_graph = other_second_program->graph();
  if (alternate_second_graph.inputs.size() != 2u ||
      other_second_graph.inputs.size() != 2u ||
      alternate_second_graph.outputs.size() != 1u ||
      other_second_graph.outputs.size() != 1u) {
    return 68;
  }
  const auto &alternate_second_input =
      alternate_second_graph.resources[alternate_second_graph.inputs[1u] - 1u];
  const auto &other_second_input =
      other_second_graph.resources[other_second_graph.inputs[1u] - 1u];
  const auto &alternate_second_output =
      alternate_second_graph
          .resources[alternate_second_graph.outputs.front() - 1u];
  const auto &other_second_output =
      other_second_graph.resources[other_second_graph.outputs.front() - 1u];
  if (alternate_second_input.integer_bits != 17u ||
      alternate_second_input.fraction_bits != 15u ||
      other_second_input.integer_bits != 18u ||
      other_second_input.fraction_bits != 14u ||
      alternate_second_output.rounding != rund::compute::Rounding::Down ||
      alternate_second_output.overflow != rund::compute::Overflow::Wrap ||
      other_second_output.rounding != rund::compute::Rounding::Down ||
      other_second_output.overflow != rund::compute::Overflow::Wrap) {
    return 69;
  }
  return 0;
}

} // namespace rund_node_graph_services
