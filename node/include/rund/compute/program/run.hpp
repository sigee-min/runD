#pragma once

#include <rund/compute/abi/job.hpp>
#include <rund/compute/abi/observe.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<std::shared_ptr<JobState>>
run_transient(const std::shared_ptr<ProgramState> &program,
              std::span<const HostView> inputs);

[[nodiscard]] Status run_into(const std::shared_ptr<ProgramState> &program,
                              std::span<const HostView> inputs,
                              Type output_type, void *output_data,
                              std::size_t output_bytes,
                              std::size_t output_count);

template <class R>
[[nodiscard]] Result<std::vector<R>>
run_host_views(const std::shared_ptr<ProgramState> &program,
               std::span<const HostView> inputs);

template <class... R>
[[nodiscard]] Result<std::tuple<HostValueT<R>...>>
run_host_outputs(const std::shared_ptr<ProgramState> &program,
                 const std::span<const HostView> inputs) {
  auto job = run_transient(program, inputs);
  if (!job) {
    return Result<std::tuple<HostValueT<R>...>>::fail(job.reason());
  }
  std::size_t output = 0u;
  return read_schema_tuple<R...>(*job, output, std::index_sequence_for<R...>{});
}

template <class R, class... A>
[[nodiscard]] Result<std::vector<R>>
run_host(const std::shared_ptr<ProgramState> &program,
         const std::span<const A>... inputs) {
  const std::array<HostView, sizeof...(A)> views{
      HostView{inputs.data(), inputs.size(), type<A>()}...};
  return run_host_views<R>(program, views);
}

template <class R>
[[nodiscard]] Result<std::vector<R>>
run_host_views(const std::shared_ptr<ProgramState> &program,
               const std::span<const HostView> inputs) {
  static_assert(std::is_trivially_copyable_v<R>);
  try {
    std::vector<R> output(program_output_size(program, 0u));
    const Status status = run_into(program, inputs, type<R>(), output.data(),
                                   output.size() * sizeof(R), output.size());
    if (!status) {
      return Result<std::vector<R>>::fail(status.reason());
    }
    return Result<std::vector<R>>::success(std::move(output));
  } catch (const std::bad_alloc &) {
    return Result<std::vector<R>>::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
