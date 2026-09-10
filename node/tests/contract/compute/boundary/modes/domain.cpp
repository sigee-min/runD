#include "local.hpp"

namespace rund::node::test_contract::boundary_modes {

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

bool CheckDomainModes() {
  using F32 = rund::compute::Fixed<1, 31>;
  using F64 = rund::compute::Fixed<1, 63>;
  return CheckDomainFamilies<std::int32_t>() &&
         CheckDomainFamilies<std::uint32_t>() &&
         CheckDomainFamilies<std::int64_t>() &&
         CheckDomainFamilies<std::uint64_t>() && CheckDomainFamilies<F32>() &&
         CheckDomainFamilies<F64>();
}

} // namespace rund::node::test_contract::boundary_modes
