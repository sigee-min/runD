#pragma once
#include <array>
#include <memory>
#include <rund/compute/abi/job.hpp>
#include <rund/compute/abi/observe.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/telemetry.hpp>
#include <span>
#include <tuple>
#include <utility>

namespace rund {
class Session;
}

namespace rund::compute {
template <class> class Program;
template <class> class Job;
class Batch;
namespace detail {
template <std::size_t I, class... Schemas>
inline constexpr std::size_t schema_output_offset = [] {
  constexpr std::array<std::size_t, sizeof...(Schemas)> widths{
      schema_leaf_count<Schemas>...};
  std::size_t offset = 0u;
  for (std::size_t index = 0u; index < I; ++index) {
    offset += widths[index];
  }
  return offset;
}();

template <class> struct OutputReader;
template <class... Schemas> struct OutputReader<Outputs<Schemas...>> final {
  template <std::size_t I>
  [[nodiscard]] static auto read(const std::shared_ptr<JobState> &state) {
    static_assert(I < sizeof...(Schemas),
                  "compute output index is out of range");
    using Schema = std::tuple_element_t<I, std::tuple<Schemas...>>;
    std::size_t output = schema_output_offset<I, Schemas...>;
    return SchemaReader<Schema>::read(state, output);
  }

  [[nodiscard]] static Result<std::tuple<HostValueT<Schemas>...>>
  read_all(const std::shared_ptr<JobState> &state) {
    std::size_t output = 0u;
    return read_schema_tuple<Schemas...>(state, output,
                                         std::index_sequence_for<Schemas...>{});
  }
};
} // namespace detail
template <class Output, class... Inputs> class Job<Output(Inputs...)> final {
public:
  Job(const Job &) = delete;
  Job(Job &&) noexcept = default;
  Job &operator=(const Job &) = delete;
  Job &operator=(Job &&) noexcept = default;

  [[nodiscard]] Status run() { return detail::run_job(state_); }
  [[nodiscard]] Status write(const std::span<const Inputs>... inputs) {
    return detail::write_job<Inputs...>(state_, inputs...);
  }
  [[nodiscard]] auto read() const
    requires(!detail::is_outputs<Output>)
  {
    if constexpr (detail::is_bounded<Output>) {
      std::size_t output = 0u;
      return detail::SchemaReader<Output>::read(state_, output);
    } else {
      return detail::read_job<Output>(state_);
    }
  }
  template <std::size_t I>
  [[nodiscard]] auto read() const
    requires(detail::is_outputs<Output>)
  {
    return detail::OutputReader<Output>::template read<I>(state_);
  }
  [[nodiscard]] auto read_all() const
    requires(detail::is_outputs<Output>)
  {
    return detail::OutputReader<Output>::read_all(state_);
  }
  [[nodiscard]] Stats stats() const noexcept {
    return detail::job_stats(state_);
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return detail::job_memory(state_);
  }
  [[nodiscard]] MemorySnapshot
  memory_snapshot(const std::span<MemoryEntry> entries) const noexcept {
    return detail::job_memory_snapshot(state_, entries);
  }
  [[nodiscard]] WriteStats write_stats() const noexcept {
    return detail::job_write_stats(state_);
  }
  [[nodiscard]] Result<telemetry::Profile> profile() const noexcept {
    return detail::job_profile(state_);
  }

private:
  template <class> friend class Program;
  friend class Batch;
  friend class ::rund::Session;
  friend struct detail::JobAccess;
  explicit Job(std::shared_ptr<detail::JobState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<detail::JobState> state_;
};
} // namespace rund::compute
