#pragma once

#include <kernel/program/executor/callback.hpp>

#include <type_traits>

namespace rund::kernel {

template <std::size_t Rank> struct PreparedEach {
  Executor exec{};
  Space<Rank> index_space{};
  u64 units = 0u;
  u64 program_generation = 0u;
  u32 partition_count = 0u;
  bool physical_tiling_enabled = false;
  u64 physical_tile_units = 0u;
  u64 physical_tile_count = 0u;
  bool valid = false;
  const char *reason = "prepared_each_not_validated";

  [[nodiscard]] explicit operator bool() const noexcept { return valid; }

  template <typename Callback>
    requires skeleton_detail::DirectCallback<Callback>
  [[nodiscard]] SkeletonResult run(Callback &&callback) const;

  template <typename Callback>
    requires std::is_invocable_r_v<void, Callback &, const Partition &>
  [[nodiscard]] SkeletonResult run_partitions(Callback &&callback) const;
};

} // namespace rund::kernel
