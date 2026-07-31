#include "model.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace rund_node_flow_contract {

template <class T>
int RunCompositeExpression(const rund::compute::Backend backend,
                           const std::array<T, 3u> &input,
                           rund::compute::Stats &stats) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "composite-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                return record(
                    field<OpField<20>>(
                        quantize<T>(dot(value, half, half, value))),
                    field<OpField<21>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            len(value, half))),
                    field<OpField<22>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            dist(value, half, half, value))),
                    field<OpField<23>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            mean(value, half))),
                    field<OpField<24>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            var(value, half))),
                    field<OpField<25>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            corr(value, half, half, value))),
                    field<OpField<26>>(quantize<T>(lerp(value, half, half))),
                    field<OpField<27>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(smoothstep(
                            fixed_zero(value), fixed_max(value), value))),
                    field<OpField<28>>(quantize<T>(
                        bandpass(value, fixed_zero(value), fixed_max(value)))),
                    field<OpField<29>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            ratio(value, half))),
                    field<OpField<30>>(quantize<T>(sum(value, half))),
                    field<OpField<31>>(quantize<T>(centered(value, half))),
                    field<OpField<32>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            zscore(value, half, half))),
                    field<OpField<33>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            angle(AngleOp::Cosine, value, half, half, value))),
                    field<OpField<34>>(quantize<T>(deadzone(value, half))),
                    field<OpField<35>>(quantize<T>(clip(value, half))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "composite expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<16u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

template <class T>
int RunHashExpression(const rund::compute::Backend backend,
                      const std::array<T, 3u> &input,
                      rund::compute::Stats &stats) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "hash-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                return record(
                    field<OpField<40>>(quantize<T>(hash(value))),
                    field<OpField<41>>(quantize<T>(hash(HashOp::Unit, value))),
                    field<OpField<42>>(quantize<T>(noise(value, half))),
                    field<OpField<43>>(quantize<T>(noise(value, half, half))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "hash expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<4u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

template <class T>
int RunExtendedExpression(const rund::compute::Backend backend,
                          const std::array<T, 3u> &input,
                          rund::compute::Stats &stats) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "extended-expression-surface", input.size(),
              [](auto value) {
                const auto half = fixed(FixedOp::Half, value);
                const auto quarter = fixed(FixedOp::Quarter, value);
                const auto store = [](auto expression) {
                  return quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                  Approximation::Deterministic>(expression);
                };
                return record(
                    field<OpField<50>>(store(conv(
                        value, half, quarter, value, half, quarter, value, half,
                        quarter, value, half, quarter, value, half))),
                    field<OpField<51>>(store(value)),
                    field<OpField<52>>(store(standardized(
                        StandardizedOp::Cubic, value, half, quarter))),
                    field<OpField<53>>(store(standardized(
                        StandardizedOp::Quartic, value, half, quarter))),
                    field<OpField<54>>(store(
                        mean(StandardizedOp::Cubic, value, half, quarter))),
                    field<OpField<55>>(store(cov(value, half, quarter, value,
                                                 half, quarter, value, half))),
                    field<OpField<56>>(store(corr(value, half, quarter, value,
                                                  half, quarter, value, half))),
                    field<OpField<57>>(
                        store(proportion(Axis::Z, value, half, quarter))),
                    field<OpField<58>>(store(reflect(
                        Axis::Z, value, half, quarter, half, quarter, value))),
                    field<OpField<59>>(store(
                        plane(GeometryOp::Parameter, value, half, quarter, half,
                              quarter, value, value, half, quarter))),
                    field<OpField<60>>(store(
                        plane(GeometryOp::Distance, value, half, quarter, half,
                              quarter, value, value, half, quarter))),
                    field<OpField<61>>(store(plane(
                        GeometryOp::Projection, Axis::Y, value, half, quarter,
                        half, quarter, value, value, half, quarter))),
                    field<OpField<62>>(
                        store(weighted_mean(value, half, quarter, value, half,
                                            quarter, half, quarter))),
                    field<OpField<63>>(
                        store(bezier(value, half, quarter, half))),
                    field<OpField<64>>(
                        store(bezier(value, half, quarter, value, half))),
                    field<OpField<65>>(
                        store(mat(MatOp::Transpose, Axis::Z, value, half,
                                  quarter, half, quarter, value))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "extended expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<16u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

template <class T>
int RunFunctionalExpression(const rund::compute::Backend backend,
                            const std::array<T, 3u> &input,
                            rund::compute::Stats &stats) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "functional-expression-surface", input.size(),
              [](auto value) {
                const auto zero = fixed_zero(value);
                const auto half = fixed(FixedOp::Half, value);
                const auto quarter = fixed(FixedOp::Quarter, value);
                const auto truth = gt(value, zero);
                const auto store = [](auto expression) {
                  return quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                  Approximation::Deterministic>(expression);
                };
                return record(
                    field<OpField<70>>(store(zero_if(
                        predicate_or(
                            predicate_not(eq(value, value)),
                            predicate_and(ge(value, quarter), le(value, half))),
                        keep_if(all(ne(value, zero), truth), neg(value))))),
                    field<OpField<71>>(store(
                        bit_xor(bit_and(value, half), bit_or(value, quarter)))),
                    field<OpField<72>>(store(
                        bandstop(clamp_range(saturate(value), quarter, half),
                                 quarter, half))),
                    field<OpField<73>>(store(select(
                        any(in_range(value, quarter, half),
                            out_range(value, zero, half)),
                        median(absdiff(value, half), midrange(value, quarter),
                               spread(value, half, quarter)),
                        step(quarter, value)))),
                    field<OpField<74>>(store(
                        diff(DifferenceOrder::Second, quarter, value, half))),
                    field<OpField<75>>(store(rms(value, half, quarter))),
                    field<OpField<76>>(store(
                        mix(poly(value, zero, half, quarter),
                            poly_deriv(value, half, quarter), half, half))),
                    field<OpField<77>>(store(
                        aff(Axis::X, half, quarter, quarter, value, half))),
                    field<OpField<78>>(
                        store(cross(value, half, half, quarter))),
                    field<OpField<79>>(
                        store(bary(Axis::X, value, quarter, zero, zero, half,
                                   zero, zero, half))),
                    field<OpField<80>>(store(add_sat(
                        unit(Axis::X, value, half),
                        add_sat(proj(Axis::Y, value, half, half, quarter),
                                reject(Axis::X, value, half, half, quarter))))),
                    field<OpField<81>>(
                        store(line(GeometryOp::Projection, Axis::X, value,
                                   quarter, zero, zero, half, half))),
                    field<OpField<82>>(store(activation(
                        ActivationOp::HardSwish,
                        activation(ActivationOp::LeakyRelu, value, quarter)))),
                    field<OpField<83>>(
                        store(add_sat(softsign(value), huber(value, quarter)))),
                    field<OpField<84>>(
                        store(add_sat(smootherstep(zero, half, value),
                                      window(WindowOp::Triangular, value)))),
                    field<OpField<85>>(store(add_sat(
                        remap(zero, half, quarter, half,
                              snap(value, half, quarter)),
                        add_sat(fade(value), add_sat(positive_part(value),
                                                     negative_part(value)))))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "functional expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<16u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

template <class T>
int RunFixedExpression(const rund::compute::Backend backend,
                       const std::array<T, 3u> &input,
                       rund::compute::Stats &stats) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .map<T>(
              "fixed-expression-surface", input.size(),
              [](auto value) {
                return record(
                    field<OpField<0>>(quantize<T>(abs(value))),
                    field<OpField<1>>(quantize<T>(abs_magnitude(value))),
                    field<OpField<2>>(quantize<T>(sign(value))),
                    field<OpField<3>>(quantize<T>(add_sat(value, value))),
                    field<OpField<4>>(quantize<T>(sub_sat(value, value))),
                    field<OpField<5>>(
                        quantize<T>(mul_fixed_scaled(value, value))),
                    field<OpField<6>>(
                        quantize<T>(mul_unsigned_fixed(value, value))),
                    field<OpField<7>>(
                        quantize<T>(mul_add_fixed(value, value, value))),
                    field<OpField<8>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(recip(value))),
                    field<OpField<9>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(sqrt(value))),
                    field<OpField<10>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(sin(value))),
                    field<OpField<11>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(cos(value))),
                    field<OpField<12>>(
                        quantize<T, Rounding::NearestEven, Overflow::Saturate,
                                 Approximation::Deterministic>(
                            atan2(value, value))));
              })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "fixed expression compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr,
                 "fixed expression resident failed backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  auto run = job->run();
  if (!run) {
    std::fprintf(stderr, "fixed expression run failed backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(run.error().size()), run.error().data());
    return 2;
  }
  const bool reads_ok = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = job->template read<I>();
              return values && values->size() == input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<13u>{});
  if (!reads_ok) {
    return 3;
  }
  stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

int CheckExpressions(const rund::compute::Backend backend,
                     std::uint64_t &graph_hash, std::uint64_t &output_hash) {
  using namespace rund::compute;
  const std::array<Fixed<1, 31>, 3u> fixed_lane32_input{
      Fixed<1, 31>::from_raw(1 << 28), Fixed<1, 31>::from_raw(1 << 29),
      Fixed<1, 31>::from_raw(1 << 30)};
  Stats fixed_lane32_stats{};
  if (const int result =
          RunFixedExpression(backend, fixed_lane32_input, fixed_lane32_stats);
      result != 0) {
    return result;
  }
  Stats composite_lane32_stats{};
  if (const int result = RunCompositeExpression(backend, fixed_lane32_input,
                                                composite_lane32_stats);
      result != 0) {
    return 20 + result;
  }
  Stats hash_lane32_stats{};
  if (const int result =
          RunHashExpression(backend, fixed_lane32_input, hash_lane32_stats);
      result != 0) {
    return 24 + result;
  }
  Stats extended_lane32_stats{};
  if (const int result = RunExtendedExpression(backend, fixed_lane32_input,
                                               extended_lane32_stats);
      result != 0) {
    return 28 + result;
  }
  Stats functional_lane32_stats{};
  if (const int result = RunFunctionalExpression(backend, fixed_lane32_input,
                                                 functional_lane32_stats);
      result != 0) {
    return 32 + result;
  }
  const std::array<Fixed<1, 63>, 3u> fixed_lane64_input{
      Fixed<1, 63>::from_raw(std::int64_t{1} << 60u),
      Fixed<1, 63>::from_raw(std::int64_t{1} << 61u),
      Fixed<1, 63>::from_raw(std::int64_t{1} << 62u)};
  Stats fixed_lane64_stats{};
  if (const int result =
          RunFixedExpression(backend, fixed_lane64_input, fixed_lane64_stats);
      result != 0) {
    return 4 + result;
  }
  const std::array<std::uint32_t, 3u> unsigned_input{1u, 4u, 0x40000000u};
  auto unsigned_program =
      flow_on(backend)
          .map<std::uint32_t>(
              "integer-expression-surface", unsigned_input.size(),
              [](auto value) {
                return record(field<OpField<0>>(shl<1u>(value)),
                              field<OpField<1>>(shr_logical<1u>(value)),
                              field<OpField<2>>(bit_not(value)),
                              field<OpField<3>>(add_sat_unsigned(value, value)),
                              field<OpField<4>>(hash(value)),
                              field<OpField<5>>(hash(HashOp::Unit, value)));
              })
          .compile();
  if (!unsigned_program) {
    return 5;
  }
  auto unsigned_job = unsigned_program->resident(unsigned_input);
  if (!unsigned_job || !unsigned_job->run()) {
    return 6;
  }
  const bool unsigned_reads = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = unsigned_job->template read<I>();
              return values && values->size() == unsigned_input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<6u>{});
  if (!unsigned_reads) {
    return 7;
  }
  const std::array<std::int32_t, 3u> signed_input{-8, -1, 4};
  auto signed_program =
      flow_on(backend)
          .map<std::int32_t>("signed-expression-surface", signed_input.size(),
                             [](auto value) {
                               return record(
                                   field<OpField<0>>(shr_arithmetic<1u>(value)),
                                   field<OpField<1>>(abs(value)),
                                   field<OpField<2>>(add_sat(value, value)));
                             })
          .compile();
  if (!signed_program) {
    return 8;
  }
  auto signed_job = signed_program->resident(signed_input);
  if (!signed_job || !signed_job->run()) {
    return 9;
  }
  const bool signed_reads = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return (([&] {
              auto values = signed_job->template read<I>();
              return values && values->size() == signed_input.size();
            }()) &&
            ...);
  }(std::make_index_sequence<3u>{});
  if (!signed_reads) {
    return 10;
  }
  const auto unsigned_stats = unsigned_job->stats();
  const auto signed_stats = signed_job->stats();
  const std::uint64_t combined_graph =
      (fixed_lane32_stats.graph_hash * 1099511628211ull) ^
      (composite_lane32_stats.graph_hash * 16777619ull) ^
      (hash_lane32_stats.graph_hash * 2166136261ull) ^
      (extended_lane32_stats.graph_hash * 1099511627791ull) ^
      (functional_lane32_stats.graph_hash * 1099511627689ull) ^
      (fixed_lane64_stats.graph_hash * 7809847782465536323ull) ^
      (unsigned_stats.graph_hash * 1469598103934665603ull) ^
      signed_stats.graph_hash;
  const std::uint64_t combined_output =
      (fixed_lane32_stats.output_hash * 1099511628211ull) ^
      (composite_lane32_stats.output_hash * 16777619ull) ^
      (hash_lane32_stats.output_hash * 2166136261ull) ^
      (extended_lane32_stats.output_hash * 1099511627791ull) ^
      (functional_lane32_stats.output_hash * 1099511627689ull) ^
      (fixed_lane64_stats.output_hash * 7809847782465536323ull) ^
      (unsigned_stats.output_hash * 1469598103934665603ull) ^
      signed_stats.output_hash;
  if (graph_hash == 0u) {
    graph_hash = combined_graph;
    output_hash = combined_output;
  } else if (graph_hash != combined_graph || output_hash != combined_output) {
    std::fprintf(
        stderr,
        "expression parity failed backend=%u graph=%llu/%llu "
        "output=%llu/%llu lane32=%llu composite=%llu hash=%llu "
        "lane64=%llu unsigned=%llu signed=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(graph_hash),
        static_cast<unsigned long long>(combined_graph),
        static_cast<unsigned long long>(output_hash),
        static_cast<unsigned long long>(combined_output),
        static_cast<unsigned long long>(fixed_lane32_stats.output_hash),
        static_cast<unsigned long long>(composite_lane32_stats.output_hash),
        static_cast<unsigned long long>(hash_lane32_stats.output_hash),
        static_cast<unsigned long long>(fixed_lane64_stats.output_hash),
        static_cast<unsigned long long>(unsigned_stats.output_hash),
        static_cast<unsigned long long>(signed_stats.output_hash));
    return 11;
  }
  return 0;
}

} // namespace rund_node_flow_contract
