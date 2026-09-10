#pragma once

#include "../internal.hpp"

#include "../../../../device/residency/pool.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

enum class Backend : std::uint8_t { Cpu, Accelerator };

struct Context final {
  VirtualPipelineState &state;
  VirtualBacking &input;
  const VirtualRunProjection &run;
  std::uint64_t epoch{};
  std::array<bool, 2u> &prefetch_pending;
  Stats &stats;
  PreparedEpoch &prepared;
  TimelineInterval &ready;
  TimelineInterval &upload;
  bool coherent_lookahead{};
  bool &poison;
  VirtualInputReuseSeed *input_reuse{};
  Backend backend{Backend::Cpu};

  residency::Pool *pool{};
  std::size_t lane{};
  residency::PrefetchReceipt prefetched{};
  std::uint64_t host_token{};
  std::uint64_t fetched{};
  bool coherent_input_capable{};
};

[[nodiscard]] bool is_accelerator(const Context &) noexcept;
[[nodiscard]] bool release_prefetch_aliases(Context &,
                                            bool source_known = true) noexcept;
[[nodiscard]] bool cancel_prefetch_receipt(Context &,
                                           residency::PrefetchReceipt &,
                                           bool source_known = true) noexcept;
[[nodiscard]] bool wait_prefetch(residency::Pool &, std::array<bool, 2u> &,
                                 bool publish, bool source_known,
                                 Backend) noexcept;

[[nodiscard]] Status admit(Context &) noexcept;
[[nodiscard]] Status supply(Context &) noexcept;
[[nodiscard]] Status schedule_lookahead(Context &) noexcept;
[[nodiscard]] Status abort(Context &, Status) noexcept;

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail
