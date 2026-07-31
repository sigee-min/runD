#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <rund/compute/buffer.hpp>
#include <rund/compute/graph/info.hpp>
#include <rund/compute/job.hpp>
#include <rund/compute/program/run.hpp>
#include <rund/compute/run.hpp>
#include <span>
#include <tuple>
#include <utility>
#include <vector>
namespace rund::compute {
class Device;
template <class, class, class> class Flow;
template <class Signature> class Program;
namespace detail {
[[nodiscard]] Result<Run>
execute_buffers(const std::shared_ptr<ProgramState> &program,
                std::span<const std::shared_ptr<BufferState>> inputs,
                std::span<const std::shared_ptr<BufferState>> outputs);

class ProgramHandle {
public:
  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] Result<Backend> backend() const noexcept {
    return program_backend(state_);
  }
  [[nodiscard]] const ::rund::compute::graph::Info &graph() const noexcept {
    return program_graph_info(state_);
  }
  [[nodiscard]] ::rund::compute::graph::Fingerprint
  fingerprint() const noexcept {
    return graph().fingerprint;
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return program_memory(state_);
  }
  [[nodiscard]] MemorySnapshot
  memory_snapshot(const std::span<MemoryEntry> entries) const noexcept {
    return program_memory_snapshot(state_, entries);
  }

protected:
  ProgramHandle() = delete;
  explicit ProgramHandle(std::shared_ptr<ProgramState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<ProgramState> state_;
};
} // namespace detail
template <class R, class A>
  requires(!detail::is_outputs<R> && !detail::is_bounded<R>)
class Program<R(A)> final : private detail::ProgramHandle {
public:
  Program(const Program &) noexcept = default;
  Program(Program &&) noexcept = default;
  Program &operator=(const Program &) noexcept = default;
  Program &operator=(Program &&) noexcept = default;
  using detail::ProgramHandle::backend;
  using detail::ProgramHandle::fingerprint;
  using detail::ProgramHandle::graph;
  using detail::ProgramHandle::memory;
  using detail::ProgramHandle::memory_snapshot;
  using detail::ProgramHandle::operator bool;
  using detail::ProgramHandle::valid;
  [[nodiscard]] std::size_t size() const noexcept {
    return detail::program_input_size(state_, 0u);
  }
  [[nodiscard]] std::size_t output_size() const noexcept {
    return detail::program_output_size(state_, 0u);
  }
  [[nodiscard]] Result<std::vector<R>>
  run(const std::span<const A> input) const {
    if (!valid()) {
      return Result<std::vector<R>>::fail(Reason::ProgramInvalid);
    }
    if (input.size() != size()) {
      return Result<std::vector<R>>::fail(Reason::ShapeMismatch);
    }
    return detail::run_host<R, A>(state_, input);
  }
  [[nodiscard]] Result<Run> run(const Buffer<A> &input,
                                Buffer<R> &output) const {
    if (!valid()) {
      return Result<Run>::fail(Reason::ProgramInvalid);
    }
    if (input.size() != size() || output.size() != output_size()) {
      return Result<Run>::fail(Reason::ShapeMismatch);
    }
    const std::array<std::shared_ptr<detail::BufferState>, 1> inputs{
        input.state_};
    const std::array<std::shared_ptr<detail::BufferState>, 1> outputs{
        output.state_};
    return detail::execute_buffers(state_, inputs, outputs);
  }
  [[nodiscard]] Result<Job<R(A)>>
  resident(const std::span<const A> input) const {
    if (!valid()) {
      return Result<Job<R(A)>>::fail(Reason::ProgramInvalid);
    }
    auto result = detail::make_job<A>(state_, input);
    if (!result) {
      return Result<Job<R(A)>>::fail(result.reason());
    }
    return Result<Job<R(A)>>::success(Job<R(A)>{std::move(result).value()});
  }

private:
  friend struct detail::ProgramAccess;
  friend class Device;
  friend struct detail::FlowAccess;
  template <class, class, class> friend class Flow;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : detail::ProgramHandle(std::move(state)) {}
};
template <class R, class A, class B, class... Rest>
  requires(!detail::is_outputs<R> && !detail::is_bounded<R>)
class Program<R(A, B, Rest...)> final : private detail::ProgramHandle {
public:
  Program(const Program &) noexcept = default;
  Program(Program &&) noexcept = default;
  Program &operator=(const Program &) noexcept = default;
  Program &operator=(Program &&) noexcept = default;
  using detail::ProgramHandle::backend;
  using detail::ProgramHandle::fingerprint;
  using detail::ProgramHandle::graph;
  using detail::ProgramHandle::memory;
  using detail::ProgramHandle::memory_snapshot;
  using detail::ProgramHandle::operator bool;
  using detail::ProgramHandle::valid;
  [[nodiscard]] std::size_t size() const noexcept {
    return detail::program_input_size(state_, 0u);
  }
  [[nodiscard]] std::size_t input_count() const noexcept {
    return 2u + sizeof...(Rest);
  }
  [[nodiscard]] std::size_t output_size() const noexcept {
    return detail::program_output_size(state_, 0u);
  }
  [[nodiscard]] Result<std::vector<R>>
  run(const std::span<const A> first, const std::span<const B> second,
      const std::span<const Rest>... rest) const {
    if (!valid()) {
      return Result<std::vector<R>>::fail(Reason::ProgramInvalid);
    }
    const std::array<std::size_t, 2u + sizeof...(Rest)> sizes{
        first.size(), second.size(), rest.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<std::vector<R>>::fail(Reason::ShapeMismatch);
    }
    return detail::run_host<R, A, B, Rest...>(state_, first, second, rest...);
  }
  [[nodiscard]] Result<Job<R(A, B, Rest...)>>
  resident(const std::span<const A> first, const std::span<const B> second,
           const std::span<const Rest>... rest) const {
    if (!valid()) {
      return Result<Job<R(A, B, Rest...)>>::fail(Reason::ProgramInvalid);
    }
    const std::array<std::size_t, 2u + sizeof...(Rest)> sizes{
        first.size(), second.size(), rest.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<Job<R(A, B, Rest...)>>::fail(Reason::ShapeMismatch);
    }
    auto result =
        detail::make_job<A, B, Rest...>(state_, first, second, rest...);
    if (!result) {
      return Result<Job<R(A, B, Rest...)>>::fail(result.reason());
    }
    return Result<Job<R(A, B, Rest...)>>::success(
        Job<R(A, B, Rest...)>{std::move(result).value()});
  }

private:
  friend struct detail::ProgramAccess;
  friend struct detail::FlowAccess;
  template <class, class, class> friend class Flow;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : detail::ProgramHandle(std::move(state)) {}
};
template <class R, class Count, class... A>
class Program<Bounded<R, Count>(A...)> final : private detail::ProgramHandle {
public:
  Program(const Program &) noexcept = default;
  Program(Program &&) noexcept = default;
  Program &operator=(const Program &) noexcept = default;
  Program &operator=(Program &&) noexcept = default;
  using detail::ProgramHandle::backend;
  using detail::ProgramHandle::fingerprint;
  using detail::ProgramHandle::graph;
  using detail::ProgramHandle::memory;
  using detail::ProgramHandle::memory_snapshot;
  using detail::ProgramHandle::operator bool;
  using detail::ProgramHandle::valid;
  [[nodiscard]] std::size_t input_count() const noexcept {
    return sizeof...(A);
  }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return detail::program_output_size(state_, 0u);
  }
  [[nodiscard]] Result<Job<Bounded<R, Count>(A...)>>
  resident(const std::span<const A>... inputs) const {
    if (!valid()) {
      return Result<Job<Bounded<R, Count>(A...)>>::fail(Reason::ProgramInvalid);
    }
    const std::array<std::size_t, sizeof...(A)> sizes{inputs.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<Job<Bounded<R, Count>(A...)>>::fail(Reason::ShapeMismatch);
    }
    auto result = detail::make_job<A...>(state_, inputs...);
    if (!result) {
      return Result<Job<Bounded<R, Count>(A...)>>::fail(result.reason());
    }
    return Result<Job<Bounded<R, Count>(A...)>>::success(
        Job<Bounded<R, Count>(A...)>{std::move(result).value()});
  }
  [[nodiscard]] Result<std::vector<R>>
  run(const std::span<const A>... inputs) const {
    if (!valid()) {
      return Result<std::vector<R>>::fail(Reason::ProgramInvalid);
    }
    const std::array<std::size_t, sizeof...(A)> sizes{inputs.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<std::vector<R>>::fail(Reason::ShapeMismatch);
    }
    const std::array<detail::HostView, sizeof...(A)> views{
        detail::HostView{inputs.data(), inputs.size(), detail::type<A>()}...};
    auto output = detail::run_host_outputs<Bounded<R, Count>>(state_, views);
    if (!output) {
      return Result<std::vector<R>>::fail(output.reason());
    }
    return Result<std::vector<R>>::success(
        std::move(std::get<0>(output.value())));
  }

private:
  friend struct detail::ProgramAccess;
  friend struct detail::FlowAccess;
  template <class, class, class> friend class Flow;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : detail::ProgramHandle(std::move(state)) {}
};
template <class... R, class... A>
class Program<Outputs<R...>(A...)> final : private detail::ProgramHandle {
public:
  static_assert(sizeof...(R) > 1u, "use Program<R(A...)> for a single output");
  Program(const Program &) noexcept = default;
  Program(Program &&) noexcept = default;
  Program &operator=(const Program &) noexcept = default;
  Program &operator=(Program &&) noexcept = default;
  using detail::ProgramHandle::backend;
  using detail::ProgramHandle::fingerprint;
  using detail::ProgramHandle::graph;
  using detail::ProgramHandle::memory;
  using detail::ProgramHandle::memory_snapshot;
  using detail::ProgramHandle::operator bool;
  using detail::ProgramHandle::valid;
  [[nodiscard]] std::size_t input_count() const noexcept {
    return sizeof...(A);
  }
  [[nodiscard]] std::size_t output_count() const noexcept {
    return detail::schema_leaf_count<Outputs<R...>>;
  }
  template <std::size_t I>
  [[nodiscard]] std::size_t output_size() const noexcept {
    static_assert(I < detail::schema_leaf_count<Outputs<R...>>,
                  "compute output index is out of range");
    return detail::program_output_size(state_, I);
  }
  [[nodiscard]] Result<Job<Outputs<R...>(A...)>>
  resident(const std::span<const A>... inputs) const {
    if (!valid()) {
      return Result<Job<Outputs<R...>(A...)>>::fail(Reason::ProgramInvalid);
    }
    const std::array<std::size_t, sizeof...(A)> sizes{inputs.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<Job<Outputs<R...>(A...)>>::fail(Reason::ShapeMismatch);
    }
    auto result = detail::make_job<A...>(state_, inputs...);
    if (!result) {
      return Result<Job<Outputs<R...>(A...)>>::fail(result.reason());
    }
    return Result<Job<Outputs<R...>(A...)>>::success(
        Job<Outputs<R...>(A...)>{std::move(result).value()});
  }
  [[nodiscard]] Result<std::tuple<detail::HostValueT<R>...>>
  run(const std::span<const A>... inputs) const {
    if (!valid()) {
      return Result<std::tuple<detail::HostValueT<R>...>>::fail(
          Reason::ProgramInvalid);
    }
    const std::array<std::size_t, sizeof...(A)> sizes{inputs.size()...};
    if (!detail::program_input_sizes_match(state_, sizes)) {
      return Result<std::tuple<detail::HostValueT<R>...>>::fail(
          Reason::ShapeMismatch);
    }
    const std::array<detail::HostView, sizeof...(A)> views{
        detail::HostView{inputs.data(), inputs.size(), detail::type<A>()}...};
    return detail::run_host_outputs<R...>(state_, views);
  }

private:
  friend struct detail::ProgramAccess;
  friend struct detail::FlowAccess;
  template <class, class, class> friend class Flow;
  explicit Program(std::shared_ptr<detail::ProgramState> state) noexcept
      : detail::ProgramHandle(std::move(state)) {}
};
} // namespace rund::compute
