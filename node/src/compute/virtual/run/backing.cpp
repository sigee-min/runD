#include "backing.hpp"

#include "../../pipeline/run/clock.hpp"
#include "../backing.hpp"

#include <rund/counter.hpp>

#include <cstring>
#include <span>

namespace rund::compute::detail {

using ::rund::detail::counter::Accumulate;

Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          const std::uint64_t required_output_bytes) noexcept {
  const std::uint64_t input_recovery =
      VirtualBackingAccess::recovery_bytes(input);
  const std::uint64_t output_recovery =
      VirtualBackingAccess::recovery_bytes(output);
  return input_recovery == 0u && output_recovery <= required_output_bytes
             ? Status::success()
             : Status::fail(Reason::BufferPoisoned);
}

Status read_virtual_wave(VirtualBacking &backing,
                         const VirtualWaveProjection &wave,
                         const VirtualRunProjection &run,
                         ResidencyStats &stats) noexcept {
  std::memset(run.input_stage, 0,
              static_cast<std::size_t>(run.input_arena_bytes));
  const std::uint64_t started = pipeline_clock();
  const Status status = backing.read(
      wave.input_offset,
      std::span<std::byte>{run.input_stage, wave.logical_input_bytes});
  Accumulate(stats.backing_io_ns, pipeline_clock() - started);
  return status;
}

Status write_virtual_wave(VirtualBacking &backing,
                          const VirtualWaveProjection &wave,
                          const VirtualRunProjection &run,
                          ResidencyStats &stats) noexcept {
  VirtualBackingAccess::require_recovery(backing, run.active.output_bytes);
  const std::uint64_t started = pipeline_clock();
  const Status status = backing.write(
      wave.output_offset,
      std::span<const std::byte>{run.output_stage, wave.logical_output_bytes});
  Accumulate(stats.backing_io_ns, pipeline_clock() - started);
  return status;
}

void clear_virtual_recovery(VirtualBacking &backing) noexcept {
  VirtualBackingAccess::clear_recovery(backing);
}

} // namespace rund::compute::detail
