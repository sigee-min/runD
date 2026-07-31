#include "../local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::numeric {
namespace {

template <class T>
[[nodiscard]] int
CheckTransformOptionMatrix(const rund::compute::Transform options) {
  using namespace rund::compute;
  using Output = std::tuple<std::vector<T>, std::vector<T>>;
  using Raw = typename T::Raw;
  const Raw half_raw = static_cast<Raw>(Raw{1} << (T::fraction_bits - 1u));
  const T half = T::from_raw(half_raw);
  const T negative_half = T::from_raw(static_cast<Raw>(-half_raw));
  const std::array<T, 4u> real{
      T::from_raw(static_cast<Raw>(Raw{1} << T::fraction_bits)), half,
      T::zero(), negative_half};
  const std::array<T, 4u> imag{T::zero(), negative_half, half, T::zero()};
  Output reference{};
  std::uint64_t reference_graph = 0u;
  std::uint64_t reference_output = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target = on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target
            .template map<T>("numeric-transform-mode", real.size(),
                             [](auto value) { return quantize<T>(value); })
            .complex()
            .fourier(options)
            .compile();
    if (!program) {
      return 1;
    }
    auto job = program->resident(real, imag);
    if (!job || !job->run() || !job->run()) {
      return 2;
    }
    const Stats warm = job->stats();
    auto output = job->read_all();
    const Stats read = job->stats();
    if (!output || warm.pipeline_compiles != 0u ||
        warm.buffer_allocations != 0u || warm.download_events != 0u ||
        warm.graph_hash == 0u || read.output_hash == 0u) {
      return 3;
    }
    if (backend == Backend::Cpu) {
      reference = std::move(*output);
      reference_graph = warm.graph_hash;
      reference_output = read.output_hash;
    } else if (*output != reference || warm.graph_hash != reference_graph ||
               read.output_hash != reference_output) {
      return 4;
    }
  }
  return 0;
}

template <class T> [[nodiscard]] int CheckTransformModes() {
  using namespace rund::compute;
  const std::array options{
      Transform{.direction = Direction::Forward, .normalize = false},
      Transform{.direction = Direction::Forward, .normalize = true},
      Transform{.direction = Direction::Inverse, .normalize = false},
      Transform{.direction = Direction::Inverse, .normalize = true},
  };
  for (const Transform option : options) {
    if (const int result = CheckTransformOptionMatrix<T>(option); result != 0) {
      return result;
    }
  }
  return 0;
}

template <class T> [[nodiscard]] bool CheckTransformRejections() {
  using namespace rund::compute;
  const std::array<T, 0u> empty{};
  auto empty_result =
      on(Target::cpu(), empty).complex(empty).fourier().collect();
  if (empty_result ||
      empty_result.error() != "compute_transform_count_not_power_of_two") {
    std::fprintf(stderr, "numeric transform empty ok=%u code=%u reason=%.*s\n",
                 empty_result ? 1u : 0u,
                 static_cast<unsigned>(empty_result.code()),
                 static_cast<int>(empty_result.error().size()),
                 empty_result.error().data());
    return false;
  }
  const std::array<T, 3u> non_power{T::zero(), T::zero(), T::zero()};
  auto non_power_result =
      on(Target::cpu(), non_power).complex(non_power).fourier().collect();
  if (non_power_result ||
      non_power_result.error() != "compute_transform_count_not_power_of_two") {
    std::fprintf(stderr,
                 "numeric transform non-power ok=%u code=%u reason=%.*s\n",
                 non_power_result ? 1u : 0u,
                 static_cast<unsigned>(non_power_result.code()),
                 static_cast<int>(non_power_result.error().size()),
                 non_power_result.error().data());
    return false;
  }
  const std::array<T, 4u> valid{T::zero(), T::zero(), T::zero(), T::zero()};
  auto invalid_direction =
      on(Target::cpu(), valid)
          .complex(valid)
          .fourier(
              {.direction = static_cast<Direction>(0xffu), .normalize = false})
          .collect();
  if (invalid_direction ||
      invalid_direction.error() != "compute_transform_direction_unsupported") {
    std::fprintf(stderr,
                 "numeric transform direction ok=%u code=%u reason=%.*s\n",
                 invalid_direction ? 1u : 0u,
                 static_cast<unsigned>(invalid_direction.code()),
                 static_cast<int>(invalid_direction.error().size()),
                 invalid_direction.error().data());
    return false;
  }
  return true;
}

} // namespace

bool CheckTransformRejections() {
  return CheckTransformRejections<rund::compute::Fixed<16, 16>>() &&
         CheckTransformRejections<rund::compute::Fixed<20, 44>>();
}

int CheckTransform32() {
  return CheckTransformModes<rund::compute::Fixed<16, 16>>();
}

int CheckTransform64() {
  return CheckTransformModes<rund::compute::Fixed<20, 44>>();
}

} // namespace rund::node::test_contract::numeric
