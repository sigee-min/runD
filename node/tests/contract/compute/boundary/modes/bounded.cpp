#include "local.hpp"

namespace rund::node::test_contract::boundary_modes {

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

bool CheckBoundedModes() {
  return CheckBoundedFamily<std::int32_t>() &&
         CheckBoundedFamily<std::uint32_t>() &&
         CheckBoundedFamily<std::int64_t>() &&
         CheckBoundedFamily<std::uint64_t>();
}

bool CheckUnsignedModes() { return CheckUnsignedFamily(); }

} // namespace rund::node::test_contract::boundary_modes
