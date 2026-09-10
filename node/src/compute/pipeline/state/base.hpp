#pragma once

#include <rund/compute/abi/state.hpp>
#include <rund/compute/abi/resource.hpp>
#include <rund/compute/fixed.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::compute::detail {

struct BufferState;

struct PipelinePublicationStepOrdinal final {
  std::size_t value{};
};

struct PipelineLogicalOutputOrdinal final {
  std::uint32_t value{};
};

struct PipelinePhysicalOutputOrdinal final {
  std::uint32_t value{};
};

enum class PipelinePhase : unsigned char {
  Ready,
  Running,
  Poisoned,
};

enum class PipelineAccess : unsigned char {
  Read,
  Write,
};

struct BufferClaim final {
  BufferState *buffer{};
  bool write{};
  // Frozen during Pipeline preparation.  Terminal publication can distinguish
  // rollback-owned state writes in the same canonical claim pass without
  // searching every declared state pair for every failed write.
  bool transactional_state{};
  bool gated_publish{};
};

struct PipelineBinding final {
  static constexpr std::uint32_t external =
      std::numeric_limits<std::uint32_t>::max();

  std::shared_ptr<BufferState> buffer{};
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  std::size_t backing_bytes{};
  ResourceAccess access{ResourceAccess::Read};
  std::uint32_t owner{external};
  bool hidden{};
};

enum class PipelineFill : std::uint8_t {
  None,
  Ordinal,
};

struct PipelineInternal final {
  // A residency cache may prebind one Device-global physical owner. Ordinary
  // Pipeline internals leave this empty and are materialized exactly once by
  // the Pipeline allocator.
  std::shared_ptr<BufferState> owner;
  Type type{Type::U32};
  FixedFormat format{};
  std::size_t count{};
  PipelineFill fill{PipelineFill::None};
};

} // namespace rund::compute::detail
