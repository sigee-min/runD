#pragma once

#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute {

class Device;
class ProgramCache;
class FlowBuilder;

namespace detail {
struct ProgramCacheState;
}

class ProgramCache final {
public:
  struct Stats final {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t waits{};
    std::uint64_t evictions{};
    std::size_t ready_entries{};
    std::size_t in_flight{};
    std::size_t capacity{};
  };

  ProgramCache(const ProgramCache &) = delete;
  ProgramCache &operator=(const ProgramCache &) = delete;
  ProgramCache(ProgramCache &&) noexcept = default;
  ProgramCache &operator=(ProgramCache &&) noexcept = default;

  [[nodiscard]] Stats stats() const noexcept;
  void clear() noexcept;

private:
  friend Result<ProgramCache> program_cache(const Device &, std::size_t);
  friend FlowBuilder on(const Device &, const ProgramCache &) noexcept;
  explicit ProgramCache(std::shared_ptr<detail::ProgramCacheState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::ProgramCacheState> state_;
};

[[nodiscard]] Result<ProgramCache> program_cache(const Device &device,
                                                 std::size_t capacity = 64u);

[[nodiscard]] FlowBuilder on(const Device &device,
                             const ProgramCache &cache) noexcept;

} // namespace rund::compute
