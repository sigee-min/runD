#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t kTail = 255u;

struct Hash final {
  std::uint64_t graph{};
  std::uint64_t output{};
};

struct DomainEvidence final {
  Hash movement{};
  Hash transpose{};
  Hash product{};
};

struct FixedEvidence final {
  Hash transform{};
  Hash factor{};
  Hash solve{};
  Hash spectrum{};
};

template <class T> [[nodiscard]] constexpr T Zero() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::zero();
  } else {
    return T{0};
  }
}

template <class T> [[nodiscard]] constexpr T Small() {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 27u);
  } else if constexpr (std::same_as<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::int64_t{1} << 59u);
  } else {
    return T{1};
  }
}

template <class T> [[nodiscard]] constexpr T SmallProduct() {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 23u);
  } else if constexpr (std::same_as<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::int64_t{1} << 55u);
  } else {
    return T{1};
  }
}

template <class T> [[nodiscard]] constexpr T Minimum() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::min();
  } else {
    return std::numeric_limits<T>::min();
  }
}

template <class T> [[nodiscard]] constexpr T Maximum() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::max();
  } else {
    return std::numeric_limits<T>::max();
  }
}

[[nodiscard]] auto Target(const rund::compute::Backend backend) {
  using namespace rund::compute;
  return on(rund::node::test_contract::target_for(backend, 2u));
}

template <class Job> [[nodiscard]] auto Read(Job &job) {
  if constexpr (requires { job.read_all(); }) {
    return job.read_all();
  } else {
    return job.read();
  }
}

template <class Job, class Validate>
[[nodiscard]] bool CheckSuccess(Job &job, const rund::compute::Backend backend,
                                const char *const family, Hash &reference,
                                Validate &&validate) {
  if (!job.run()) {
    std::fprintf(stderr, "boundary modes run backend=%u family=%s\n",
                 static_cast<unsigned>(backend), family);
    return false;
  }
  auto output = Read(job);
  const rund::compute::Stats observed = job.stats();
  if (!output || !validate(*output) || observed.graph_hash == 0u ||
      observed.output_hash == 0u || observed.backend != backend) {
    std::fprintf(stderr, "boundary modes result backend=%u family=%s\n",
                 static_cast<unsigned>(backend), family);
    return false;
  }
  const Hash hash{.graph = observed.graph_hash, .output = observed.output_hash};
  if (reference.graph == 0u) {
    reference = hash;
    return true;
  }
  const bool same =
      reference.graph == hash.graph && reference.output == hash.output;
  if (!same) {
    std::fprintf(stderr,
                 "boundary modes parity backend=%u family=%s "
                 "graph=%llu/%llu output=%llu/%llu\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<unsigned long long>(reference.graph),
                 static_cast<unsigned long long>(hash.graph),
                 static_cast<unsigned long long>(reference.output),
                 static_cast<unsigned long long>(hash.output));
  }
  return same;
}

template <class T> [[nodiscard]] std::vector<T> BoundaryValues() {
  std::vector<T> values(kTail, Zero<T>());
  values[0u] = Minimum<T>();
  values[1u] = Maximum<T>();
  return values;
}

template <class T>
[[nodiscard]] bool CheckMovement(const rund::compute::Backend backend,
                                 Hash &evidence) {
  using namespace rund::compute;
  const std::vector<T> input = BoundaryValues<T>();
  std::vector<std::uint32_t> order(kTail);
  std::vector<std::uint32_t> flags(kTail);
  for (std::size_t index = 0u; index < kTail; ++index) {
    order[index] = static_cast<std::uint32_t>(kTail - index - 1u);
    flags[index] = static_cast<std::uint32_t>(index & 1u);
  }
  auto program =
      Target(backend)
          .template input<T>(kTail)
          .template zip_input<std::uint32_t>(kTail)
          .template zip_input<std::uint32_t>(kTail)
          .branch([](auto values, auto indices, auto mask) {
            return outputs(values.gather(indices), values.scatter(indices),
                           values.partition(mask));
          })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input, order, flags);
  if (!job) {
    return false;
  }
  std::vector<T> reversed(input.rbegin(), input.rend());
  std::vector<T> partitioned;
  partitioned.reserve(kTail);
  for (const std::uint32_t flag : {0u, 1u}) {
    for (std::size_t index = 0u; index < kTail; ++index) {
      if (flags[index] == flag) {
        partitioned.push_back(input[index]);
      }
    }
  }
  return CheckSuccess(*job, backend, "movement-tail", evidence,
                      [&](const auto &output) {
                        return std::get<0>(output) == reversed &&
                               std::get<1>(output) == reversed &&
                               std::get<2>(output) == partitioned;
                      });
}

template <class T>
[[nodiscard]] bool CheckMatrix(const rund::compute::Backend backend,
                               Hash &transpose_evidence,
                               Hash &product_evidence) {
  using namespace rund::compute;
  const std::vector<T> input = BoundaryValues<T>();
  auto transpose = Target(backend)
                       .template map<T>("boundary-transpose", kTail,
                                        [](auto value) { return value; })
                       .matrix({1u, kTail, 1u})
                       .transpose()
                       .compile();
  if (!transpose) {
    return false;
  }
  auto transpose_job = transpose->resident(input);
  if (!transpose_job ||
      !CheckSuccess(*transpose_job, backend, "transpose-tail",
                    transpose_evidence,
                    [&](const auto &output) { return output == input; })) {
    return false;
  }

  std::vector<T> left(kTail, Zero<T>());
  std::vector<T> right(kTail, Small<T>());
  left[0u] = Small<T>();
  auto product = Target(backend)
                     .template map<T>("boundary-matmul", kTail,
                                      [](auto value) { return value; })
                     .matrix({1u, kTail, 1u})
                     .matmul({kTail, 1u, 1u})
                     .compile();
  if (!product) {
    return false;
  }
  auto product_job = product->resident(left, right);
  const std::vector<T> expected{SmallProduct<T>()};
  return product_job &&
         CheckSuccess(*product_job, backend, "matmul-tail", product_evidence,
                      [&](const auto &output) { return output == expected; });
}

template <class T> [[nodiscard]] bool CheckDomainFamilies() {
  DomainEvidence evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckMovement<T>(backend, evidence.movement) ||
        !CheckMatrix<T>(backend, evidence.transpose, evidence.product)) {
      std::fprintf(stderr, "boundary modes domain backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

template <class T>
[[nodiscard]] bool CheckBounded(const rund::compute::Backend backend,
                                Hash &evidence) {
  using namespace rund::compute;
  std::vector<T> input(kTail);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<T>(index & 1u);
  }
  std::vector<T> right(kTail, T{2});
  right[0u] = T{1};
  auto program =
      Target(backend)
          .template input<T>(kTail)
          .template zip_input<T>(right.size())
          .branch([](auto values, auto matches) {
            auto filtered =
                values.filter([](auto value) { return value != T{0}; });
            auto expanded = filtered.expand(
                MaxItems{1u}, [](auto value) { return value; },
                [](auto value, auto) { return value; });
            auto grouped = filtered.group_by([](auto value) { return value; })
                               .aggregate([](auto group) {
                                 return record(group.key(), group.count());
                               });
            auto joined = filtered.join(
                MaxMatches{1u}, matches, [](auto value) { return value; },
                [](auto value) { return value; },
                [](auto left, auto matched) { return left + matched; });
            return outputs(filtered, expanded, grouped, joined);
          })
          .compile();
  if (!program) {
    const auto reason = program.error();
    std::fprintf(stderr, "boundary modes bounded compile backend=%u: %.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  auto job = program->resident(input, right);
  if (!job) {
    const auto reason = job.error();
    std::fprintf(stderr, "boundary modes bounded resident backend=%u: %.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  const std::vector<T> ones(kTail / 2u, T{1});
  const std::vector<T> twos(kTail / 2u, T{2});
  return CheckSuccess(
      *job, backend, "bounded-tail", evidence, [&](const auto &output) {
        const auto &groups = std::get<2>(output);
        const bool valid = std::get<0>(output) == ones &&
                           std::get<1>(output) == ones &&
                           std::get<0>(groups) == std::vector<T>{T{1}} &&
                           std::get<1>(groups) ==
                               std::vector<std::uint32_t>{
                                   static_cast<std::uint32_t>(kTail / 2u)} &&
                           std::get<3>(output) == twos;
        if (!valid) {
          std::fprintf(stderr,
                       "boundary modes bounded shape backend=%u "
                       "filtered=%zu expanded=%zu group_keys=%zu "
                       "group_counts=%zu joined=%zu\n",
                       static_cast<unsigned>(backend),
                       std::get<0>(output).size(), std::get<1>(output).size(),
                       std::get<0>(groups).size(), std::get<1>(groups).size(),
                       std::get<3>(output).size());
        }
        return valid;
      });
}

template <class T> [[nodiscard]] bool CheckBoundedFamily() {
  Hash evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckBounded<T>(backend, evidence)) {
      std::fprintf(stderr, "boundary modes bounded backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckUnsigned(const rund::compute::Backend backend,
                                 Hash &evidence) {
  using namespace rund::compute;
  std::vector<std::uint32_t> input(kTail);
  std::vector<std::uint32_t> selected;
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::uint32_t>(index % 4u);
    if (input[index] != 0u) {
      selected.push_back(static_cast<std::uint32_t>(index));
    }
  }
  auto program =
      Target(backend)
          .template input<std::uint32_t>(kTail)
          .branch([](auto values) {
            return rund::compute::outputs(values.compact({.capacity = kTail}),
                                          values.histogram({.bins = 4u}));
          })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  const std::vector<std::uint32_t> histogram{64u, 64u, 64u, 63u};
  return job && CheckSuccess(*job, backend, "unsigned-tail", evidence,
                             [&](const auto &output) {
                               return std::get<0>(output) == selected &&
                                      std::get<1>(output) == histogram;
                             });
}

[[nodiscard]] bool CheckUnsignedFamily() {
  Hash evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckUnsigned(backend, evidence)) {
      std::fprintf(stderr, "boundary modes unsigned backend=%u\n",
                   static_cast<unsigned>(backend));
      return false;
    }
  }
  return true;
}

template <class T>
[[nodiscard]] bool CheckFixed(const rund::compute::Backend backend,
                              FixedEvidence &evidence) {
  using namespace rund::compute;
  auto invalid = Target(backend)
                     .template map<T>("boundary-transform-invalid", kTail,
                                      [](auto value) { return value; })
                     .complex()
                     .fourier()
                     .compile();
  if (invalid ||
      invalid.error() != "compute_transform_count_not_power_of_two") {
    return false;
  }

  std::vector<T> signal(kTail + 1u, Zero<T>());
  signal[0u] = Small<T>();
  auto transform = Target(backend)
                       .template map<T>("boundary-transform", signal.size(),
                                        [](auto value) { return value; })
                       .complex()
                       .fourier()
                       .compile();
  if (!transform) {
    return false;
  }
  auto transform_job = transform->resident(signal, signal);
  const std::vector<T> transformed(signal.size(), Small<T>());
  if (!transform_job ||
      !CheckSuccess(*transform_job, backend, "transform-boundary",
                    evidence.transform, [&](const auto &output) {
                      return std::get<0>(output) == transformed &&
                             std::get<1>(output) == transformed;
                    })) {
    return false;
  }

  std::vector<T> matrices(kTail, Small<T>());
  auto factor = Target(backend)
                    .template map<T>("boundary-factor", kTail,
                                     [](auto value) { return value; })
                    .matrix({1u, 1u, kTail})
                    .lu()
                    .compile();
  auto solve = Target(backend)
                   .template map<T>("boundary-solve", kTail,
                                    [](auto value) { return value; })
                   .matrix({1u, 1u, kTail})
                   .lu()
                   .solve(1u)
                   .compile();
  auto spectrum = Target(backend)
                      .template map<T>("boundary-spectrum", kTail,
                                       [](auto value) { return value; })
                      .matrix({1u, 1u, kTail})
                      .template svd<SpectrumVectors::Values>()
                      .compile();
  if (!factor || !solve || !spectrum) {
    return false;
  }
  auto factor_job = factor->resident(matrices);
  auto solve_job = solve->resident(matrices, matrices);
  auto spectrum_job = spectrum->resident(matrices);
  const std::vector<std::uint32_t> zero_status(kTail, 0u);
  if (!factor_job ||
      !CheckSuccess(*factor_job, backend, "factor-tail", evidence.factor,
                    [&](const auto &output) {
                      return std::get<0>(output) == matrices &&
                             std::get<1>(output) == zero_status &&
                             std::get<2>(output) == zero_status;
                    }) ||
      !solve_job ||
      !CheckSuccess(*solve_job, backend, "solve-tail", evidence.solve,
                    [&](const auto &output) {
                      return std::get<0>(output).size() == kTail &&
                             std::get<1>(output) == zero_status;
                    }) ||
      !spectrum_job ||
      !CheckSuccess(*spectrum_job, backend, "spectrum-tail", evidence.spectrum,
                    [&](const auto &output) {
                      return std::get<0>(output) == matrices &&
                             std::get<1>(output) == zero_status;
                    })) {
    return false;
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckFixedFamily() {
  FixedEvidence evidence{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckFixed<T>(backend, evidence)) {
      std::fprintf(stderr, "boundary modes fixed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

} // namespace

int RunComputeBoundaryModesContract() {
  using F32 = rund::compute::Fixed<1, 31>;
  using F64 = rund::compute::Fixed<1, 63>;

  // Collective 255-tail, empty, extrema, and overflow semantics are owned by
  // compute.collective-modes and are intentionally not recompiled here.
  if (!CheckDomainFamilies<std::int32_t>() ||
      !CheckDomainFamilies<std::uint32_t>() ||
      !CheckDomainFamilies<std::int64_t>() ||
      !CheckDomainFamilies<std::uint64_t>() || !CheckDomainFamilies<F32>() ||
      !CheckDomainFamilies<F64>()) {
    return 1;
  }
  if (!CheckBoundedFamily<std::int32_t>() ||
      !CheckBoundedFamily<std::uint32_t>() ||
      !CheckBoundedFamily<std::int64_t>() ||
      !CheckBoundedFamily<std::uint64_t>()) {
    return 2;
  }
  if (!CheckUnsignedFamily()) {
    return 3;
  }
  return CheckFixedFamily<F32>() && CheckFixedFamily<F64>() ? 0 : 4;
}
