#include "local.hpp"

#include <rund/compute/math.hpp>

#include <cstdio>
#include <vector>

namespace rund_node_test_pipeline {

[[nodiscard]] bool FailureEvidence(Pipeline &pipeline, const Backend backend,
                                   const Reason reason,
                                   const std::uint64_t status_entries) {
  const Stats stats = pipeline.stats();
  const std::uint64_t control_commands = backend == Backend::Cpu ? 0u : 4u;
  return pipeline.poisoned() && stats.backend == backend &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == 1u &&
         stats.pipeline.status_entry_count == status_entries &&
         stats.command_submits == (backend == Backend::Cpu ? 0u : 1u) &&
         (backend == Backend::Cpu ? stats.pipeline.control_byte_count == 0u
                                  : stats.pipeline.control_byte_count == 128u) &&
         stats.pipeline.control_command_count == control_commands &&
         pipeline.run().reason() == Reason::PipelinePoisoned &&
         reason != Reason::Ok;
}

[[nodiscard]] int CheckSemanticStatus(rund::compute::Device &device,
                                      const Backend backend) {
  using namespace rund::compute;
  using Real = Fixed<20, 44>;
  constexpr auto scalar = [](const std::int64_t value) {
    return Real::from_raw(value << Real::fraction_bits);
  };
  constexpr Real zero = Real::zero();
  constexpr Real one = scalar(1);
  constexpr Real two = scalar(2);
  constexpr Real four = scalar(4);
  constexpr std::array<Real, 4u> singular{one, two, two, four};
  constexpr std::array<Real, 2u> rhs{one, two};
  constexpr std::array<Real, 4u> spectrum{one, one, one, two};

  auto pass = on(device)
                  .map<Real>("pipeline-status-pass", 4u,
                             [](auto value) { return quantize<Real>(value); })
                  .compile();
  auto factor = on(device)
                    .map<Real>("pipeline-status-factor", 4u,
                               [](auto value) { return quantize<Real>(value); })
                    .matrix<2u, 2u>()
                    .lu()
                    .compile();
  if (!pass || !factor) {
    return 1;
  }
  auto raw_factor = factor->resident(singular);
  if (!raw_factor || !raw_factor->run()) {
    return 2;
  }
  const auto raw_factor_status = raw_factor->template read<2u>();
  if (!raw_factor_status ||
      *raw_factor_status != std::vector<std::uint32_t>{1u}) {
    return 3;
  }
  auto factor_input = Upload(device, singular);
  auto factor_middle = device.buffer<Real>(4u);
  auto factor_values = device.buffer<Real>(4u);
  auto factor_pivots = device.buffer<std::uint32_t>(2u);
  auto factor_status = device.buffer<std::uint32_t>(1u);
  if (!factor_input || !factor_middle || !factor_values || !factor_pivots ||
      !factor_status) {
    return 4;
  }
  auto factor_pipeline =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .then(*pass, read(*factor_input), write(*factor_middle))
          .then(*factor, read(*factor_middle),
                write(*factor_values, *factor_pivots, *factor_status))
          .prepare();
  if (!factor_pipeline) {
    return 5;
  }
  const Status factor_result = factor_pipeline->run();
  std::array<PipelineStepProfile, 2u> factor_rows{};
  const auto factor_profile = factor_pipeline->profile(factor_rows);
  if (factor_result || factor_result.reason() != Reason::FactorSingular ||
      !FailureEvidence(*factor_pipeline, backend, factor_result.reason(), 1u) ||
      !factor_profile ||
      factor_profile->execution.pipeline.verified_step_count != 1u ||
      factor_profile->execution.pipeline.failed_step_index != 1u ||
      !factor_rows[0].execution.available() ||
      !factor_rows[1].execution.available() ||
      !ProfileMemoryReconciles(*factor_profile, factor_rows)) {
    return 6;
  }

  auto solve = on(device)
                   .map<Real>("pipeline-status-solve", 4u,
                              [](auto value) { return quantize<Real>(value); })
                   .matrix<2u, 2u>()
                   .template solve<FactorOp::Lu, 1u>()
                   .compile();
  if (!solve) {
    return 7;
  }
  auto raw_solve = solve->resident(singular, rhs);
  if (!raw_solve || !raw_solve->run()) {
    return 8;
  }
  const auto raw_solve_status = raw_solve->template read<1u>();
  if (!raw_solve_status ||
      *raw_solve_status != std::vector<std::uint32_t>{1u}) {
    return 9;
  }
  auto solve_input = Upload(device, singular);
  auto solve_rhs = Upload(device, rhs);
  auto solve_middle = device.buffer<Real>(4u);
  auto solve_values = device.buffer<Real>(2u);
  auto solve_status = device.buffer<std::uint32_t>(1u);
  if (!solve_input || !solve_rhs || !solve_middle || !solve_values ||
      !solve_status) {
    return 10;
  }
  auto solve_pipeline =
      pipeline(device)
          .then(*pass, read(*solve_input), write(*solve_middle))
          .then(*solve, read(*solve_middle, *solve_rhs),
                write(*solve_values, *solve_status))
          .prepare();
  if (!solve_pipeline) {
    return 11;
  }
  const Status solve_result = solve_pipeline->run();
  if (solve_result || solve_result.reason() != Reason::SolveSingular ||
      !FailureEvidence(*solve_pipeline, backend, solve_result.reason(), 1u)) {
    return 12;
  }

  auto spectrum_program =
      on(device)
          .map<Real>("pipeline-status-spectrum", 4u,
                     [](auto value) { return quantize<Real>(value); })
          .matrix<2u, 2u>()
          .template eigen<SpectrumVectors::Values>(1u)
          .compile();
  if (!spectrum_program) {
    return 13;
  }
  auto raw_spectrum = spectrum_program->resident(spectrum);
  if (!raw_spectrum || !raw_spectrum->run()) {
    return 14;
  }
  const auto raw_spectrum_status = raw_spectrum->template read<1u>();
  if (!raw_spectrum_status ||
      *raw_spectrum_status != std::vector<std::uint32_t>{1u}) {
    return 15;
  }
  auto spectrum_input = Upload(device, spectrum);
  auto spectrum_middle = device.buffer<Real>(4u);
  auto spectrum_values = device.buffer<Real>(2u);
  auto spectrum_status = device.buffer<std::uint32_t>(1u);
  if (!spectrum_input || !spectrum_middle || !spectrum_values ||
      !spectrum_status) {
    return 16;
  }
  auto spectrum_pipeline =
      pipeline(device)
          .then(*pass, read(*spectrum_input), write(*spectrum_middle))
          .then(*spectrum_program, read(*spectrum_middle),
                write(*spectrum_values, *spectrum_status))
          .prepare();
  if (!spectrum_pipeline) {
    return 17;
  }
  const Status spectrum_result = spectrum_pipeline->run();
  const bool spectrum_evidence = FailureEvidence(*spectrum_pipeline, backend,
                                                 spectrum_result.reason(), 1u);
  if (spectrum_result ||
      spectrum_result.reason() != Reason::SpectrumNonConvergence ||
      !spectrum_evidence) {
    const Stats stats = spectrum_pipeline->stats();
    std::fprintf(
        stderr,
        "pipeline spectrum result=%d reason=%u evidence=%d poisoned=%d "
        "verified=%llu failed=%llu entries=%llu submits=%llu "
        "control_bytes=%llu "
        "control_commands=%llu rerun=%u\n",
        static_cast<int>(static_cast<bool>(spectrum_result)),
        static_cast<unsigned>(spectrum_result.reason()),
        static_cast<int>(spectrum_evidence),
        static_cast<int>(spectrum_pipeline->poisoned()),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.status_entry_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.pipeline.control_byte_count),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned>(spectrum_pipeline->run().reason()));
    return 18;
  }
  // Metal Pipeline dependencies must cross a real encoder visibility
  // boundary, not rely on the scheduling of concurrent ICB commands. This
  // exact one-iteration spectrum is a sensitive RAW witness: stale zeroed
  // input incorrectly reports convergence.
  if (backend == Backend::Metal) {
    for (std::size_t attempt = 0u; attempt < 16u; ++attempt) {
      auto middle = device.buffer<Real>(4u);
      auto values = device.buffer<Real>(2u);
      auto status = device.buffer<std::uint32_t>(1u);
      if (!middle || !values || !status) {
        return 19;
      }
      auto witness =
          pipeline(device)
              .then(*pass, read(*spectrum_input), write(*middle))
              .then(*spectrum_program, read(*middle), write(*values, *status))
              .prepare();
      if (!witness) {
        return 20;
      }
      const Status result = witness->run();
      if (result || result.reason() != Reason::SpectrumNonConvergence ||
          !FailureEvidence(*witness, backend, result.reason(), 1u)) {
        return 21;
      }
    }
  }
  if (backend != Backend::Cpu) {
    constexpr std::size_t alias_steps = 16u;
    constexpr std::array<Real, 4u> identity{one, zero, zero, one};
    auto first = Upload(device, singular);
    auto second = Upload(device, identity);
    auto values = device.buffer<Real>(4u);
    auto pivots = device.buffer<std::uint32_t>(2u);
    auto status = device.buffer<std::uint32_t>(1u);
    if (!first || !second || !values || !pivots || !status) {
      return 22;
    }
    auto alias_builder = pipeline(device);
    for (std::size_t index = 0u; index < alias_steps; ++index) {
      alias_builder.then(*factor, read((index & 1u) == 0u ? *first : *second),
                         write(*values, *pivots, *status));
    }
    auto aliased = std::move(alias_builder).prepare();
    if (!aliased) {
      return 23;
    }
    const Status result = aliased->run();
    const Stats stats = aliased->stats();
    const std::uint64_t expected_commands = 2u * alias_steps + 2u;
    if (result || result.reason() != Reason::FactorSingular ||
        stats.pipeline.verified_step_count != 0u ||
        stats.pipeline.failed_step_index != 0u ||
        stats.pipeline.status_entry_count != alias_steps ||
        stats.pipeline.control_byte_count != 128u ||
        stats.pipeline.control_command_count != expected_commands) {
      std::fprintf(
          stderr,
          "pipeline alias backend=%u reason=%u verified=%llu failed=%llu "
          "entries=%llu control-bytes=%llu control-commands=%llu "
          "expected=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(result.reason()),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(stats.pipeline.status_entry_count),
          static_cast<unsigned long long>(stats.pipeline.control_byte_count),
          static_cast<unsigned long long>(stats.pipeline.control_command_count),
          static_cast<unsigned long long>(expected_commands));
      return 24;
    }
  }

  constexpr std::array<std::int64_t, 16u> gather_source{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  constexpr std::array<std::uint32_t, 8u> first_indices{0, 1, 2, 3,
                                                        4, 5, 6, 16};
  constexpr std::array<std::uint32_t, 8u> later_indices{0, 16, 2, 3,
                                                        4, 5,  6, 7};
  auto gather = on(device)
                    .input<std::int64_t>(gather_source.size())
                    .zip_input<std::uint32_t>(first_indices.size())
                    .branch([](auto values, auto indices) {
                      return values.gather(indices);
                    })
                    .compile();
  auto gather_values = Upload(device, gather_source);
  auto first_index = Upload(device, first_indices);
  auto later_index = Upload(device, later_indices);
  auto first_output = device.buffer<std::int64_t>(first_indices.size());
  auto later_output = device.buffer<std::int64_t>(later_indices.size());
  if (!gather || !gather_values || !first_index || !later_index ||
      !first_output || !later_output) {
    return 25;
  }
  auto ordered_failure = pipeline(device)
                             .then(*gather, read(*gather_values, *first_index),
                                   write(*first_output))
                             .then(*gather, read(*gather_values, *later_index),
                                   write(*later_output))
                             .prepare();
  if (!ordered_failure) {
    return 26;
  }
  const Status ordered_status = ordered_failure->run();
  const Stats ordered_stats = ordered_failure->stats();
  if (ordered_status ||
      ordered_status.reason() != Reason::GatherIndexOutOfRange ||
      ordered_stats.pipeline.verified_step_count != 0u ||
      ordered_stats.pipeline.failed_step_index != 0u ||
      ordered_stats.control.overflow_ordinal != 7u) {
    std::fprintf(stderr,
                 "pipeline ordered failure backend=%u reason=%u verified=%llu "
                 "failed=%llu overflow=%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(ordered_status.reason()),
                 static_cast<unsigned long long>(
                     ordered_stats.pipeline.verified_step_count),
                 static_cast<unsigned long long>(
                     ordered_stats.pipeline.failed_step_index),
                 static_cast<unsigned long long>(
                     ordered_stats.control.overflow_ordinal));
    return 27;
  }
  (void)zero;
  return 0;
}

} // namespace rund_node_test_pipeline
