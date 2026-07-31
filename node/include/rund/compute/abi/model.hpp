#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <rund/compute/fixed.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/status.hpp>
#include <tuple>
#include <type_traits>
#include <vector>
namespace rund::compute {
namespace graph {
struct Info;
}
enum class Scan : unsigned char;
enum class Reduce : unsigned char;
enum class FactorOp : unsigned char;
enum class SpectrumOp : unsigned char;
enum class SpectrumVectors : unsigned char;
template <class... T> struct Outputs final {};
template <class T> struct Scalar final {};
template <class Tag, class T> struct Field final {};
template <class... T> struct Record final {};
template <class T>
using CountFor =
    std::conditional_t<sizeof(T) == 8u, std::uint64_t, std::uint32_t>;
template <class T, class Count = CountFor<T>> struct Bounded final {};
namespace detail {
template <class T> inline constexpr bool is_outputs = false;
template <class... T> inline constexpr bool is_outputs<Outputs<T...>> = true;
template <class T> inline constexpr bool is_bounded = false;
template <class T, class Count>
inline constexpr bool is_bounded<Bounded<T, Count>> = true;
template <class T> struct BoundedTraits;
template <class T, class Count> struct BoundedTraits<Bounded<T, Count>> final {
  using Value = T;
  using CountType = Count;
};
template <class T> inline constexpr std::size_t schema_leaf_count = 1u;
template <class T>
inline constexpr std::size_t schema_leaf_count<Scalar<T>> = 1u;
template <class T, class Count>
inline constexpr std::size_t schema_leaf_count<Bounded<T, Count>> = 2u;
template <class Tag, class T>
inline constexpr std::size_t schema_leaf_count<Field<Tag, T>> =
    schema_leaf_count<T>;
template <class... T>
inline constexpr std::size_t schema_leaf_count<Record<T...>> =
    (schema_leaf_count<T> + ... + 0u);
template <class... T>
inline constexpr std::size_t schema_leaf_count<Outputs<T...>> =
    (schema_leaf_count<T> + ... + 0u);
template <class T> struct HostValue final {
  using Type = std::vector<T>;
};
template <class T> struct HostValue<Scalar<T>> final {
  using Type = T;
};
template <class T, class Count> struct HostValue<Bounded<T, Count>> final {
  using Type = std::vector<T>;
};
template <class Tag, class T> struct HostValue<Field<Tag, T>> final {
  using Type = typename HostValue<T>::Type;
};
template <class... T> struct HostValue<Record<T...>> final {
  using Type = std::tuple<typename HostValue<T>::Type...>;
};
template <class T> using HostValueT = typename HostValue<T>::Type;
struct BufferState;
class CompileService;
struct DeviceState;
struct ExprState;
struct FlowAccess;
struct FlowState;
struct GraphState;
struct JobAccess;
struct JobState;
struct PipelineBuildState;
struct PipelineState;
struct BufferAccess;
struct DeviceAccess;
struct ProgramAccess;
struct ProgramCacheState;
struct ProgramState;
struct RunState;
struct StateSnapshotState;
inline constexpr std::size_t MaxOutputs = 16u;
inline constexpr std::size_t MaxMapInputs = 16u;
struct ValueIds final {
  [[nodiscard]] constexpr bool empty() const noexcept { return count == 0u; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }
  [[nodiscard]] constexpr std::uint32_t front() const noexcept {
    return values.front();
  }
  [[nodiscard]] constexpr std::uint32_t
  operator[](const std::size_t index) const noexcept {
    return values[index];
  }
  [[nodiscard]] constexpr const std::uint32_t *begin() const noexcept {
    return values.data();
  }
  [[nodiscard]] constexpr const std::uint32_t *end() const noexcept {
    return values.data() + count;
  }
  constexpr void push_back(const std::uint32_t value) noexcept {
    values[count++] = value;
  }

private:
  std::array<std::uint32_t, MaxOutputs> values{};
  std::uint32_t count{};
};
static_assert(std::is_trivially_copyable_v<ValueIds>);
static_assert(sizeof(ValueIds) == sizeof(std::uint32_t) * (MaxOutputs + 1u));
enum class ExprOp : unsigned char {
  Input,
  Constant,
  Index,
  Add,
  Subtract,
  Multiply,
  MultiplyWrap,
  Divide,
  Negate,
  BitAnd,
  BitOr,
  BitXor,
  Min,
  Max,
  Clamp,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  PredicateNot,
  PredicateAnd,
  PredicateOr,
  Mask,
  Select,
  Abs,
  AbsMagnitude,
  Sign,
  BitNot,
  AddSat,
  AddSatUnsigned,
  SubSat,
  NegPositiveFixed,
  MulFixed,
  MulFixedScaled,
  MulUnsignedFixed,
  MulAddFixed,
  Reciprocal,
  Sqrt,
  Rsqrt,
  Sin,
  Cos,
  Tan,
  Exp,
  Log,
  Atan2,
  ShiftLeft,
  ShiftRightLogical,
  ShiftRightArithmetic,
  Quantize,
  CheckedOrdinal,
  BoundaryMask,
};
struct ExprRef final {
  std::shared_ptr<ExprState> state;
  std::uint32_t node{};
  Type type{Type::I32};
  FixedFormat fixed_format{};
};
enum class Primitive : unsigned char {
  SegmentedScan,
  SegmentedReduce,
  Sort,
  Argsort,
  Compact,
  Gather,
  Histogram,
  Partition,
  Reduce,
  Scatter,
  Stencil,
  Transform,
  Matrix,
  Factor,
  Solve,
  Spectrum,
  ScatterReduce,
};

[[nodiscard]] constexpr Reason
primitive_execution_reason(const Primitive primitive,
                           const std::uint32_t status) noexcept {
  if (status == 0u) {
    return Reason::Ok;
  }
  switch (primitive) {
  case Primitive::Factor:
    switch (status) {
    case 1u:
      return Reason::FactorSingular;
    case 2u:
      return Reason::FactorNotPositiveDefinite;
    case 3u:
      return Reason::FactorPivotUnderflow;
    case 4u:
      return Reason::FactorScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  case Primitive::Solve:
    switch (status) {
    case 1u:
      return Reason::SolveSingular;
    case 2u:
      return Reason::SolveNotPositiveDefinite;
    case 3u:
      return Reason::SolvePivotUnderflow;
    case 4u:
      return Reason::SolveScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  case Primitive::Spectrum:
    switch (status) {
    case 1u:
      return Reason::SpectrumNonConvergence;
    case 2u:
      return Reason::SpectrumScalingInvalid;
    default:
      return Reason::ReasonInvalid;
    }
  default:
    return Reason::ReasonInvalid;
  }
}

struct PrimitiveOptions final {
  std::uint64_t first{};
  std::uint64_t second{};
  std::uint64_t third{};
  std::uint64_t fourth{};
  std::uint32_t mode{};
  std::uint32_t extra{};
  bool flag{};
};
struct GraphArg final {
  std::uint32_t value{};
  Type type{Type::I32};
  std::size_t count{};
  FixedFormat fixed_format{};
};
struct GraphOut final {
  std::uint32_t value{};
  Type type{Type::I32};
  std::size_t count{};
  FixedFormat fixed_format{};
  std::vector<GraphArg> outputs;
};
struct HostView final {
  const void *data{};
  std::size_t count{};
  Type type{Type::I32};
};
enum class ResourceAccess : unsigned char { Read, Write };
struct ResourceView final {
  std::shared_ptr<BufferState> buffer;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  ResourceAccess access{ResourceAccess::Read};
};
struct BoundedIds final {
  std::uint32_t values{};
  std::uint32_t count{};
};
struct BoundedInputSchema final {
  std::uint32_t count{};
  std::size_t capacity{};
};
// Canonical resident execution control attached to an otherwise ordinary
// functional Flow step. Values are logical Flow resource ids; no backend
// handle or host callback enters graph identity.
struct FlowControl final {
  std::uint32_t count{};
  std::uint32_t predicate{};
  std::size_t capacity{};
  std::uint64_t predicate_expected{};
  std::uint32_t iteration{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return count == 0u && predicate == 0u;
  }
};
struct ComplexIds final {
  std::uint32_t real{};
  std::uint32_t imag{};
};
struct SolveIds final {
  std::uint32_t values{};
  std::uint32_t status{};
};
struct FactorIds final {
  std::uint32_t packed{};
  std::uint32_t pivots{};
  std::uint32_t status{};
};
struct SpectrumIds final {
  std::uint32_t values{};
  std::uint32_t vectors{};
  std::uint32_t status{};
};
} // namespace detail
} // namespace rund::compute
