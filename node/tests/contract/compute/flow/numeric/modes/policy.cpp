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

template <class T> [[nodiscard]] int CheckWindowPolicyMatrix() {
  using namespace rund::compute;
  using Output = std::tuple<std::vector<T>, std::vector<T>, std::vector<T>,
                            std::vector<T>>;
  using Raw = typename T::Raw;
  const Raw half_raw = static_cast<Raw>(Raw{1} << (T::fraction_bits - 1u));
  const std::array<T, 4u> input{T::from_raw(half_raw),
                                T::from_raw(static_cast<Raw>(-half_raw)),
                                T::max(), T::min()};
  Output reference{};
  std::uint64_t reference_graph = 0u;
  std::uint64_t reference_output = 0u;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target = on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target.template input<T>(input.size())
            .branch([](auto values) {
              const auto nearest =
                  values.map("numeric-policy-nearest", [](auto value) {
                    return quantize<T, Rounding::NearestEven,
                                    Overflow::Saturate, Approximation::Exact>(
                        value * value);
                  });
              const auto down =
                  values.map("numeric-policy-down", [](auto value) {
                    return quantize<T, Rounding::Down, Overflow::Wrap,
                                    Approximation::Exact>(value * value);
                  });
              const auto up = values.map("numeric-policy-up", [](auto value) {
                return quantize<T, Rounding::Up, Overflow::Saturate,
                                Approximation::Exact>(value * value);
              });
              const auto deterministic =
                  values.map("numeric-policy-deterministic", [](auto value) {
                    return quantize<T, Rounding::TowardZero, Overflow::Wrap,
                                    Approximation::Deterministic>(value *
                                                                  value);
                  });
              constexpr WindowSpec clipped{
                  .op = Window::Sum,
                  .radius = 1u,
                  .edge = WindowEdge::Clip,
              };
              return outputs(nearest.window(clipped), down.window(clipped),
                             up.window(clipped), deterministic.window(clipped));
            })
            .compile();
    if (!program) {
      std::fprintf(
          stderr, "numeric window backend=%u compile code=%u reason=%.*s\n",
          static_cast<unsigned>(backend), static_cast<unsigned>(program.code()),
          static_cast<int>(program.error().size()), program.error().data());
      return 1;
    }
    auto job = program->resident(input);
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

template <class T>
[[nodiscard]] int
CheckStoredPolicyPropagation(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  const Raw one_raw = static_cast<Raw>(Raw{1} << T::fraction_bits);
  const T zero = T::zero();
  const T one = T::from_raw(one_raw);
  const T two = T::from_raw(static_cast<Raw>(one_raw * Raw{2}));
  const T negative_one = T::from_raw(static_cast<Raw>(-one_raw));
  const std::array<T, 4u> input{zero, one, two, negative_one};
  const std::vector<T> expected{one, two, negative_one};

  const auto policy_chain = [&](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected), input)
        .map("stored-policy-source",
             [](auto value) {
               return quantize<T, Rounding::Down, Overflow::Wrap,
                               Approximation::Exact>(value);
             })
        .branch([](auto values) {
          const auto mapped = values.map("stored-policy-map", [](auto value) {
            return quantize<T, Rounding::Down, Overflow::Wrap,
                            Approximation::Exact>(value);
          });
          const auto combined = mapped.combine(
              "stored-policy-combine", values, [](auto left, auto right) {
                return quantize<T, Rounding::Down, Overflow::Wrap,
                                Approximation::Exact>(left + right);
              });
          return zip(mapped, combined)
              .map("stored-policy-zip",
                   [](auto left, auto right) {
                     return quantize<T, Rounding::Down, Overflow::Wrap,
                                     Approximation::Exact>(right - left);
                   })
              .filter([](auto value) { return value != T::zero(); });
        })
        .map("stored-policy-bounded-map",
             [](auto value) {
               return quantize<T, Rounding::Down, Overflow::Wrap,
                               Approximation::Exact>(value);
             })
        .collect();
  };
  const auto backend_policy = policy_chain(backend);
  if (!backend_policy || *backend_policy != expected) {
    return 1;
  }

  const std::array<T, 4u> imag{zero, negative_one, one, zero};
  const auto complex_real = [&](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected), input)
        .map("stored-policy-complex-source",
             [](auto value) {
               return quantize<T, Rounding::Down, Overflow::Wrap,
                               Approximation::Exact>(value);
             })
        .complex(imag)
        .real()
        .collect();
  };
  const auto backend_real = complex_real(backend);
  const std::vector<T> real_expected{input.begin(), input.end()};
  if (!backend_real || *backend_real != real_expected) {
    return 2;
  }

  const std::array<T, 4u> identity{one, zero, zero, one};
  const std::array<T, 4u> rhs{one, two, negative_one, zero};
  const auto direct_solve = [&](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected), identity)
        .map("stored-policy-direct-matrix",
             [](auto value) {
               return quantize<T, Rounding::Down, Overflow::Wrap,
                               Approximation::Deterministic>(value);
             })
        .template matrix<2u, 2u>()
        .solve(rhs, FactorOp::Lu, 2u)
        .values()
        .collect();
  };
  const auto backend_solve = direct_solve(backend);
  const std::vector<T> solve_expected{rhs.begin(), rhs.end()};
  if (!backend_solve || *backend_solve != solve_expected) {
    return 3;
  }
  return 0;
}

template <class T>
[[nodiscard]] bool
HasDerivedPolicy(const rund::compute::graph::Info &info) noexcept {
  using namespace rund::compute;
  bool found = false;
  for (const graph::Resource &resource : info.resources) {
    if (resource.type != graph::Value::Fixed ||
        resource.visibility == graph::Visibility::Input) {
      continue;
    }
    found = true;
    if (resource.integer_bits != T::integer_bits ||
        resource.fraction_bits != T::fraction_bits ||
        resource.element_bytes != sizeof(T) ||
        resource.rounding != Rounding::Down ||
        resource.overflow != Overflow::Wrap ||
        resource.approximation != Approximation::Deterministic) {
      return false;
    }
  }
  return found;
}

template <class T> [[nodiscard]] int CheckNondefaultPolicyGraphMatrix() {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  const Raw one_raw = static_cast<Raw>(Raw{1} << T::fraction_bits);
  const T zero = T::zero();
  const T one = T::from_raw(one_raw);
  const T two = T::from_raw(static_cast<Raw>(one_raw * Raw{2}));
  const T negative_one = T::from_raw(static_cast<Raw>(-one_raw));
  const std::array<T, 4u> values{one, zero, two, negative_one};
  const std::array<T, 4u> impulse{one, zero, zero, zero};
  const std::array<T, 4u> imaginary{zero, zero, zero, zero};
  std::uint64_t reduce_graph = 0u;
  std::uint64_t reduce_output = 0u;
  std::uint64_t transform_graph = 0u;
  std::uint64_t transform_output = 0u;

  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto reduce_target =
        on(rund::node::test_contract::target_for(backend, 2u));
    auto reduced =
        reduce_target
            .template map<T>(
                "nondefault-policy-reduce", values.size(),
                [](auto value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap,
                                  Approximation::Deterministic>(value);
                })
            .filter([](auto value) { return value != T::zero(); })
            .reduce(Reduce::Sum)
            .map("nondefault-policy-scalar",
                 [](auto value) {
                   return quantize<T, Rounding::Down, Overflow::Wrap,
                                   Approximation::Deterministic>(value);
                 })
            .compile();
    if (!reduced || !HasDerivedPolicy<T>(reduced->graph())) {
      return 1;
    }
    std::size_t maps = 0u;
    std::size_t partitions = 0u;
    std::size_t reductions = 0u;
    for (const graph::Node &node : reduced->graph().nodes) {
      maps += node.operation == graph::Operation::Map ? 1u : 0u;
      partitions += node.operation == graph::Operation::Partition ? 1u : 0u;
      reductions += node.operation == graph::Operation::Reduce ? 1u : 0u;
    }
    if (maps < 2u || partitions == 0u || reductions < 2u) {
      return 2;
    }
    auto reduce_job = reduced->resident(values);
    if (!reduce_job || !reduce_job->run()) {
      return 3;
    }
    auto reduced_value = reduce_job->read();
    const Stats reduced_stats = reduce_job->stats();
    if (!reduced_value || *reduced_value != std::vector<T>{two} ||
        reduced_stats.graph_hash == 0u || reduced_stats.output_hash == 0u) {
      return 4;
    }
    if (backend == Backend::Cpu) {
      reduce_graph = reduced_stats.graph_hash;
      reduce_output = reduced_stats.output_hash;
    } else if (reduced_stats.graph_hash != reduce_graph ||
               reduced_stats.output_hash != reduce_output) {
      return 5;
    }

    const auto transform_target =
        on(rund::node::test_contract::target_for(backend, 2u));
    auto transformed =
        transform_target
            .template map<T>(
                "nondefault-policy-transform", impulse.size(),
                [](auto value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap,
                                  Approximation::Deterministic>(value);
                })
            .complex()
            .fourier()
            .real()
            .compile();
    if (!transformed || !HasDerivedPolicy<T>(transformed->graph())) {
      return 6;
    }
    maps = 0u;
    std::size_t transforms = 0u;
    for (const graph::Node &node : transformed->graph().nodes) {
      maps += node.operation == graph::Operation::Map ? 1u : 0u;
      transforms += node.operation == graph::Operation::Transform ? 1u : 0u;
    }
    if (maps == 0u || transforms != 1u) {
      return 7;
    }
    auto transform_job = transformed->resident(impulse, imaginary);
    if (!transform_job) {
      std::fprintf(stderr,
                   "numeric policy transform backend=%u resident code=%u "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<unsigned>(transform_job.code()),
                   static_cast<int>(transform_job.error().size()),
                   transform_job.error().data());
      return 8;
    }
    const Status ran = transform_job->run();
    if (!ran) {
      std::fprintf(stderr,
                   "numeric policy transform backend=%u run code=%u "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<unsigned>(ran.code()),
                   static_cast<int>(ran.error().size()), ran.error().data());
      return 8;
    }
    auto real = transform_job->read();
    const Stats transform_stats = transform_job->stats();
    if (!real || *real != std::vector<T>{one, one, one, one} ||
        transform_stats.graph_hash == 0u || transform_stats.output_hash == 0u) {
      return 9;
    }
    if (backend == Backend::Cpu) {
      transform_graph = transform_stats.graph_hash;
      transform_output = transform_stats.output_hash;
    } else if (transform_stats.graph_hash != transform_graph ||
               transform_stats.output_hash != transform_output) {
      return 10;
    }
  }
  return 0;
}
} // namespace

int CheckWindow32() {
  return CheckWindowPolicyMatrix<rund::compute::Fixed<16, 16>>();
}

int CheckWindow64() {
  return CheckWindowPolicyMatrix<rund::compute::Fixed<20, 44>>();
}

int CheckStored32(const rund::compute::Backend backend) {
  return CheckStoredPolicyPropagation<rund::compute::Fixed<16, 16>>(backend);
}

int CheckStored64(const rund::compute::Backend backend) {
  return CheckStoredPolicyPropagation<rund::compute::Fixed<20, 44>>(backend);
}

int CheckGraph32() {
  return CheckNondefaultPolicyGraphMatrix<rund::compute::Fixed<16, 16>>();
}

int CheckGraph64() {
  return CheckNondefaultPolicyGraphMatrix<rund::compute::Fixed<20, 44>>();
}

} // namespace rund::node::test_contract::numeric
