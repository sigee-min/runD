#include "test/assert.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <class T, std::size_t N>
std::uint64_t GraphHash(const std::array<T, N> &input) {
  auto program =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<T>("domain-identity", input.size(),
                  [](auto value) {
                    if constexpr (rund::compute::detail::FixedValue<T>) {
                      return rund::compute::quantize<T>(value + value);
                    } else {
                      return value + value;
                    }
                  })
          .compile();
  if (!program) {
    std::fprintf(stderr, "numeric domain %u compile failed: %.*s\n",
                 static_cast<unsigned>(rund::compute::detail::type<T>()),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 0u;
  }
  TEST_ASSERT(program);
  auto job = program->resident(std::span<const T>{input});
  TEST_ASSERT(job);
  TEST_ASSERT(job->run());
  return job->stats().graph_hash;
}

template <class T, std::size_t N>
std::uint64_t ScanGraphHash(const std::array<T, N> &input) {
  auto program = rund::compute::on(rund::compute::Target::cpu(1u))
                     .template map<T>(
                         "domain-scan", input.size(),
                         [](auto value) {
                           if constexpr (rund::compute::detail::FixedValue<T>) {
                             return rund::compute::quantize<T>(value);
                           } else {
                             return value;
                           }
                         })
                     .scan(rund::compute::Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return 0u;
  }
  auto job = program->resident(std::span<const T>{input});
  if (!job || !job->run() || !job->read()) {
    return 0u;
  }
  return job->stats().graph_hash;
}

template <class T, std::size_t N>
std::vector<T> DivideByTwo(const std::array<T, N> &input) {
  auto output = rund::compute::on(rund::compute::Target::cpu(1u), input)
                    .map("integer-divide", [](auto value) { return value / 2; })
                    .collect();
  if (!output) {
    std::fprintf(stderr, "integer divide failed: %.*s\n",
                 static_cast<int>(output.error().size()),
                 output.error().data());
    return {};
  }
  return std::move(output).value();
}

template <class T>
std::string_view DivideZeroReason(const std::uint32_t workers) {
  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<T> input(kCount, T{7});
  auto program = rund::compute::on(rund::compute::Target::cpu(workers))
                     .map<T>("integer-divide-zero", input.size(),
                             [](auto value) { return value / 0; })
                     .compile();
  if (!program) {
    return program.error();
  }
  auto job = program->resident(std::span<const T>{input});
  if (!job) {
    return job.error();
  }
  const auto status = job->run();
  return status ? std::string_view{"unexpected_success"} : status.error();
}

template <class T>
std::string_view DivideOverflowReason(const std::uint32_t workers) {
  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<T> input(kCount, std::numeric_limits<T>::min());
  auto program = rund::compute::on(rund::compute::Target::cpu(workers))
                     .map<T>("integer-divide-overflow", input.size(),
                             [](auto value) { return value / -1; })
                     .compile();
  if (!program) {
    return program.error();
  }
  auto job = program->resident(std::span<const T>{input});
  if (!job) {
    return job.error();
  }
  const auto status = job->run();
  return status ? std::string_view{"unexpected_success"} : status.error();
}

template <class T> [[nodiscard]] T FromSmall(const std::int64_t value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(static_cast<std::int32_t>(value));
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(value);
  } else {
    return static_cast<T>(value);
  }
}

template <class T> [[nodiscard]] std::uint64_t BitsOf(const T value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return static_cast<std::uint32_t>(value.raw());
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return static_cast<std::uint64_t>(value.raw());
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<std::uint32_t>(value);
  } else {
    return static_cast<std::uint64_t>(value);
  }
}

struct ScanObservation final {
  std::vector<std::uint64_t> values;
  rund::compute::Stats stats;
};

template <class T>
[[nodiscard]] bool ObserveScan(const std::uint32_t workers,
                               ScanObservation &observation) {
  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<T> input(kCount);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if constexpr (std::is_unsigned_v<T>) {
      input[index] = FromSmall<T>(static_cast<std::int64_t>(index % 5u));
    } else {
      input[index] = FromSmall<T>(static_cast<std::int64_t>(index % 5u) - 2);
    }
  }
  auto program = rund::compute::on(rund::compute::Target::cpu(workers))
                     .template map<T>(
                         "numeric-cross-tile-scan", input.size(),
                         [](auto value) {
                           if constexpr (rund::compute::detail::FixedValue<T>) {
                             return rund::compute::quantize<T>(value);
                           } else {
                             return value;
                           }
                         })
                     .scan(rund::compute::Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(std::span<const T>{input});
  if (!job || !job->run()) {
    return false;
  }
  auto output = job->read();
  if (!output) {
    return false;
  }
  observation.values.reserve(output->size());
  for (const T value : *output) {
    observation.values.push_back(BitsOf(value));
  }
  observation.stats = job->stats();
  return true;
}

template <class T>
[[nodiscard]] bool CheckScanParity(const std::uint32_t workers) {
  ScanObservation one{};
  ScanObservation many{};
  if (!ObserveScan<T>(1u, one) || !ObserveScan<T>(workers, many)) {
    return false;
  }
  return one.values == many.values &&
         one.stats.graph_hash == many.stats.graph_hash &&
         one.stats.output_hash == many.stats.output_hash &&
         one.stats.worker_count == 1u && many.stats.worker_count == workers &&
         many.stats.tile_count > 1u &&
         (workers == 1u || many.stats.participating_workers > 1u);
}

} // namespace

int RunComputeNumericContract() {
  using rund::compute::Fixed;

  static_assert(std::is_trivially_copyable_v<Fixed<1, 31>>);
  static_assert(std::is_trivially_copyable_v<Fixed<1, 63>>);
  static_assert(sizeof(Fixed<1, 31>) == sizeof(std::int32_t));
  static_assert(sizeof(Fixed<1, 63>) == sizeof(std::int64_t));
  static_assert(!std::is_constructible_v<Fixed<1, 31>, float>);
  static_assert(!std::is_constructible_v<Fixed<1, 31>, double>);
  static_assert(!std::is_constructible_v<Fixed<1, 63>, float>);
  static_assert(!std::is_constructible_v<Fixed<1, 63>, double>);
  static_assert(rund::compute::detail::type<std::int32_t>() ==
                rund::compute::detail::Type::I32);
  static_assert(rund::compute::detail::type<std::uint32_t>() ==
                rund::compute::detail::Type::U32);
  static_assert(rund::compute::detail::type<std::int64_t>() ==
                rund::compute::detail::Type::I64);
  static_assert(rund::compute::detail::type<std::uint64_t>() ==
                rund::compute::detail::Type::U64);
  static_assert(rund::compute::detail::type<Fixed<1, 31>>() ==
                rund::compute::detail::Type::FixedLane32);
  static_assert(rund::compute::detail::type<Fixed<1, 63>>() ==
                rund::compute::detail::Type::FixedLane64);

  constexpr Fixed<1, 31> narrow = Fixed<1, 31>::from_raw(-1234567);
  constexpr Fixed<1, 63> wide = Fixed<1, 63>::from_raw(-1234567890123456ll);
  TEST_ASSERT(narrow.raw() == -1234567);
  TEST_ASSERT(wide.raw() == -1234567890123456ll);

  using Fixed16x16 = Fixed<16, 16>;
  const std::array fixed16x16{Fixed16x16::from_raw(98304),
                              Fixed16x16::from_raw(-98304), Fixed16x16::max()};
  auto squared16x16 =
      rund::compute::on(rund::compute::Target::cpu(1u), fixed16x16)
          .map("fixed-16-16-square",
               [](auto value) {
                 return rund::compute::quantize<Fixed16x16>(value * value);
               })
          .collect();
  if (!squared16x16) {
    std::fprintf(stderr, "fixed 16.16 square failed: %.*s\n",
                 static_cast<int>(squared16x16.error().size()),
                 squared16x16.error().data());
  }
  TEST_ASSERT(squared16x16);
  TEST_ASSERT((*squared16x16)[0].raw() == 147456);
  TEST_ASSERT((*squared16x16)[1].raw() == 147456);
  TEST_ASSERT((*squared16x16)[2].raw() ==
              std::numeric_limits<std::int32_t>::max());

  const std::array fused16x16{Fixed16x16::max(), Fixed16x16::min(),
                              Fixed16x16::from_raw(98304),
                              Fixed16x16::from_raw(-98304)};
  auto fused16x16_output =
      rund::compute::on(rund::compute::Target::cpu(1u), fused16x16)
          .map("fixed-16-16-multiply-add-carry",
               [](auto value) {
                 return rund::compute::quantize<Fixed16x16>(
                     rund::compute::mul_add_fixed(value, value, value));
               })
          .collect();
  TEST_ASSERT(fused16x16_output);
  TEST_ASSERT((*fused16x16_output)[0] == Fixed16x16::max());
  TEST_ASSERT((*fused16x16_output)[1] == Fixed16x16::max());
  TEST_ASSERT((*fused16x16_output)[2].raw() == 245760);
  TEST_ASSERT((*fused16x16_output)[3].raw() == 49152);

  using WideStorage = Fixed<32, 32>;
  const std::array fused_wide{WideStorage::from_raw(6442450944ll),
                              WideStorage::from_raw(-6442450944ll)};
  auto fused_wide_output =
      rund::compute::on(rund::compute::Target::cpu(1u), fused_wide)
          .map("fixed-wide-multiply-add-tight-bound",
               [](auto value) {
                 return rund::compute::quantize<WideStorage>(
                     rund::compute::mul_add_fixed(value, value, value));
               })
          .collect();
  TEST_ASSERT(fused_wide_output);
  TEST_ASSERT((*fused_wide_output)[0].raw() == 16106127360ll);
  TEST_ASSERT((*fused_wide_output)[1].raw() == 3221225472ll);

  const std::array half16x16{Fixed16x16::from_raw(32768)};
  auto implicit_nonlinear =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed16x16>(
              "fixed-implicit-nonlinear", half16x16.size(),
              [](auto value) { return rund::compute::sqrt(value + value); })
          .compile();
  TEST_ASSERT(!implicit_nonlinear);
  if (implicit_nonlinear.error() != "compute_fixed_quantize_required") {
    std::fprintf(stderr, "implicit nonlinear reason: %.*s\n",
                 static_cast<int>(implicit_nonlinear.error().size()),
                 implicit_nonlinear.error().data());
  }
  TEST_ASSERT(implicit_nonlinear.error() == "compute_fixed_quantize_required");
  auto implicit_add =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed16x16>("fixed-implicit-add", half16x16.size(),
                           [](auto value) { return value + value; })
          .compile();
  TEST_ASSERT(!implicit_add);
  TEST_ASSERT(implicit_add.error() == "compute_fixed_quantize_required");
  auto first_error =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed16x16>("fixed-first-error", half16x16.size(),
                           [](auto value) {
                             return rund::compute::sqrt(value + value) + value;
                           })
          .compile();
  TEST_ASSERT(!first_error);
  TEST_ASSERT(first_error.error() == "compute_fixed_quantize_required");
  auto approximation_downgrade =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed16x16>("fixed-approximation-downgrade", half16x16.size(),
                           [](auto value) {
                             return rund::compute::quantize<Fixed16x16>(
                                 rund::compute::sqrt(value));
                           })
          .compile();
  TEST_ASSERT(!approximation_downgrade);
  TEST_ASSERT(approximation_downgrade.error() ==
              "compute_fixed_approximation_downgrade");
  auto fixed_policy_mismatch =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed16x16>("fixed-policy-mismatch-rejected", 1u,
                           [](auto value) {
                             const auto down_wrap = rund::compute::quantize<
                                 Fixed16x16, rund::compute::Rounding::Down,
                                 rund::compute::Overflow::Wrap>(value);
                             const auto up_saturate = rund::compute::quantize<
                                 Fixed16x16, rund::compute::Rounding::Up,
                                 rund::compute::Overflow::Saturate>(value);
                             return rund::compute::quantize<Fixed16x16>(
                                 down_wrap + up_saturate);
                           })
          .compile();
  TEST_ASSERT(!fixed_policy_mismatch);
  TEST_ASSERT(fixed_policy_mismatch.error() == "compute_fixed_format_mismatch");
  auto explicit_nonlinear =
      rund::compute::on(rund::compute::Target::cpu(1u), half16x16)
          .map("fixed-explicit-nonlinear",
               [](auto value) {
                 return rund::compute::quantize<
                     Fixed16x16, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Saturate,
                     rund::compute::Approximation::Deterministic>(
                     rund::compute::sqrt(
                         rund::compute::quantize<Fixed16x16>(value + value)));
               })
          .collect();
  TEST_ASSERT(explicit_nonlinear);
  TEST_ASSERT((*explicit_nonlinear)[0].raw() == 65536);

  using FixedLane32x32 = Fixed<32, 32>;
  const std::array fixed_lane32x32{FixedLane32x32::from_raw(6442450944ll),
                                   FixedLane32x32::from_raw(-6442450944ll)};
  auto precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<FixedLane32x32>("fixed-129-bit-add-rejected",
                               fixed_lane32x32.size(),
                               [](auto value) {
                                 return rund::compute::quantize<FixedLane32x32>(
                                     value * value + value);
                               })
          .compile();
  TEST_ASSERT(!precision_capacity);
  TEST_ASSERT(precision_capacity.error() == "compute_fixed_precision_capacity");
  using Fixed1x63 = Fixed<1, 63>;
  auto select_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed1x63>(
              "fixed-129-bit-select-rejected", 1u,
              [](auto value) {
                const auto high_fraction = value * value;
                const auto high_integer = (value + value) + value;
                return rund::compute::quantize<Fixed1x63>(rund::compute::select(
                    value == value, high_fraction, high_integer));
              })
          .compile();
  TEST_ASSERT(!select_precision_capacity);
  TEST_ASSERT(select_precision_capacity.error() ==
              "compute_fixed_precision_capacity");
  auto clamp_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed1x63>("fixed-129-bit-clamp-rejected", 1u,
                          [](auto value) {
                            const auto high_fraction = value * value;
                            const auto high_integer = (value + value) + value;
                            return rund::compute::quantize<Fixed1x63>(
                                rund::compute::clamp(
                                    high_integer, high_fraction, high_integer));
                          })
          .compile();
  TEST_ASSERT(!clamp_precision_capacity);
  TEST_ASSERT(clamp_precision_capacity.error() ==
              "compute_fixed_precision_capacity");
  auto accumulated32x32 =
      rund::compute::on(rund::compute::Target::cpu(1u), fixed_lane32x32)
          .map("fixed-lane64-multiply-add",
               [](auto value) {
                 const auto squared =
                     rund::compute::quantize<FixedLane32x32>(value * value);
                 return rund::compute::quantize<FixedLane32x32>(squared +
                                                                value);
               })
          .collect();
  if (!accumulated32x32) {
    std::fprintf(stderr, "Fixed<32,32> multiply-add failed: %.*s\n",
                 static_cast<int>(accumulated32x32.error().size()),
                 accumulated32x32.error().data());
  }
  TEST_ASSERT(accumulated32x32);
  TEST_ASSERT((*accumulated32x32)[0].raw() == 16106127360ll);
  TEST_ASSERT((*accumulated32x32)[1].raw() == 3221225472ll);

  const std::array fixed_lane32{
      Fixed<1, 31>::from_raw(-7), Fixed<1, 31>::from_raw(0),
      Fixed<1, 31>::from_raw(0x40000000), Fixed<1, 31>::from_raw(0x7fffffff)};
  auto fixed_lane32_output =
      rund::compute::on(rund::compute::Target::cpu(1u), fixed_lane32)
          .map("fixed_lane32-identity",
               [](auto value) {
                 return rund::compute::quantize<Fixed<1, 31>>(value);
               })
          .collect();
  if (!fixed_lane32_output) {
    std::fprintf(stderr, "fixed lane32 identity failed: %.*s\n",
                 static_cast<int>(fixed_lane32_output.error().size()),
                 fixed_lane32_output.error().data());
  }
  TEST_ASSERT(fixed_lane32_output);
  TEST_ASSERT(fixed_lane32_output->size() == fixed_lane32.size());
  for (std::size_t index = 0; index < fixed_lane32.size(); ++index) {
    TEST_ASSERT((*fixed_lane32_output)[index].raw() ==
                fixed_lane32[index].raw());
  }

  const std::array fixed_lane64{Fixed<1, 63>::from_raw(-9),
                                Fixed<1, 63>::from_raw(0),
                                Fixed<1, 63>::from_raw(0x4000000000000000ll),
                                Fixed<1, 63>::from_raw(0x7fffffffffffffffll)};
  auto fixed_lane64_output =
      rund::compute::on(rund::compute::Target::cpu(1u), fixed_lane64)
          .map("fixed_lane64-identity",
               [](auto value) {
                 return rund::compute::quantize<Fixed<1, 63>>(value);
               })
          .collect();
  TEST_ASSERT(fixed_lane64_output);
  TEST_ASSERT(fixed_lane64_output->size() == fixed_lane64.size());
  for (std::size_t index = 0; index < fixed_lane64.size(); ++index) {
    TEST_ASSERT((*fixed_lane64_output)[index].raw() ==
                fixed_lane64[index].raw());
  }

  const std::array<std::uint64_t, 6> hashes{
      GraphHash(std::array<std::int32_t, 4>{1, 2, 3, 4}),
      GraphHash(std::array<std::uint32_t, 4>{1, 2, 3, 4}),
      GraphHash(std::array<std::int64_t, 4>{1, 2, 3, 4}),
      GraphHash(std::array<std::uint64_t, 4>{1, 2, 3, 4}),
      GraphHash(fixed_lane32),
      GraphHash(fixed_lane64)};
  for (std::size_t left = 0; left < hashes.size(); ++left) {
    TEST_ASSERT(hashes[left] != 0u);
    for (std::size_t right = left + 1u; right < hashes.size(); ++right) {
      TEST_ASSERT(hashes[left] != hashes[right]);
    }
  }
  const std::array<std::uint64_t, 6> scan_hashes{
      ScanGraphHash(std::array<std::int32_t, 4>{1, 2, 3, 4}),
      ScanGraphHash(std::array<std::uint32_t, 4>{1, 2, 3, 4}),
      ScanGraphHash(std::array<std::int64_t, 4>{1, 2, 3, 4}),
      ScanGraphHash(std::array<std::uint64_t, 4>{1, 2, 3, 4}),
      ScanGraphHash(
          std::array{Fixed<1, 31>::from_raw(1), Fixed<1, 31>::from_raw(2),
                     Fixed<1, 31>::from_raw(3), Fixed<1, 31>::from_raw(4)}),
      ScanGraphHash(
          std::array{Fixed<1, 63>::from_raw(1), Fixed<1, 63>::from_raw(2),
                     Fixed<1, 63>::from_raw(3), Fixed<1, 63>::from_raw(4)})};
  for (std::size_t left = 0; left < scan_hashes.size(); ++left) {
    TEST_ASSERT(scan_hashes[left] != 0u);
    for (std::size_t right = left + 1u; right < scan_hashes.size(); ++right) {
      TEST_ASSERT(scan_hashes[left] != scan_hashes[right]);
    }
  }
  TEST_ASSERT(DivideByTwo(std::array<std::int32_t, 4>{-9, -1, 8, 11}) ==
              std::vector<std::int32_t>({-4, 0, 4, 5}));
  TEST_ASSERT(DivideByTwo(std::array<std::uint32_t, 4>{0, 1, 8, 11}) ==
              std::vector<std::uint32_t>({0, 0, 4, 5}));
  TEST_ASSERT(DivideByTwo(std::array<std::int64_t, 4>{-9, -1, 8, 11}) ==
              std::vector<std::int64_t>({-4, 0, 4, 5}));
  TEST_ASSERT(DivideByTwo(std::array<std::uint64_t, 4>{0, 1, 8, 11}) ==
              std::vector<std::uint64_t>({0, 0, 4, 5}));

  const unsigned available = std::thread::hardware_concurrency();
  const std::uint32_t workers =
      std::max(1u, std::min(available == 0u ? 1u : available, 4u));
  TEST_ASSERT(CheckScanParity<std::int32_t>(workers));
  TEST_ASSERT(CheckScanParity<std::uint32_t>(workers));
  TEST_ASSERT(CheckScanParity<std::int64_t>(workers));
  TEST_ASSERT(CheckScanParity<std::uint64_t>(workers));
  TEST_ASSERT((CheckScanParity<Fixed<1, 31>>(workers)));
  TEST_ASSERT((CheckScanParity<Fixed<1, 63>>(workers)));
  for (const std::uint32_t width : {1u, workers}) {
    TEST_ASSERT(DivideZeroReason<std::int32_t>(width) ==
                "compute_integer_divide_by_zero");
    TEST_ASSERT(DivideZeroReason<std::uint64_t>(width) ==
                "compute_integer_divide_by_zero");
    const auto overflow32 = DivideOverflowReason<std::int32_t>(width);
    const auto overflow64 = DivideOverflowReason<std::int64_t>(width);
    if (overflow32 != "compute_integer_divide_overflow" ||
        overflow64 != "compute_integer_divide_overflow") {
      std::fprintf(
          stderr, "integer divide overflow width=%u i32=%.*s i64=%.*s\n", width,
          static_cast<int>(overflow32.size()), overflow32.data(),
          static_cast<int>(overflow64.size()), overflow64.data());
    }
    TEST_ASSERT(overflow32 == "compute_integer_divide_overflow");
    TEST_ASSERT(overflow64 == "compute_integer_divide_overflow");
  }
  return 0;
}
