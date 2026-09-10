#pragma once
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <rund/compute/abi/flow.hpp>
#include <rund/compute/cache.hpp>
#include <rund/compute/device.hpp>
#include <rund/compute/expr/recipe.hpp>
#include <rund/compute/expr/select.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/program.hpp>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
namespace rund::compute {
enum class FactorOp : unsigned char;
enum class SpectrumOp : unsigned char;
enum class SpectrumVectors : unsigned char;
namespace stage {
inline constexpr std::size_t Dynamic = static_cast<std::size_t>(-1);
struct Exact final {};
template <class Count> struct Bounded final {};
template <class Key, class Count> struct Grouped final {};
struct Complex final {};
template <std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Matrix final {};
struct Scalar final {};
template <FactorOp Op, std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Factor final {};
template <std::size_t Rows = Dynamic, std::size_t Cols = Dynamic,
          std::size_t Batches = Dynamic>
struct Solve final {};
template <SpectrumOp Op, SpectrumVectors V, std::size_t Rows = Dynamic,
          std::size_t Cols = Dynamic, std::size_t Batches = Dynamic>
struct Spectrum final {};
} // namespace stage
namespace input {
// Flow type identity only; neither marker is stored in a Flow instance.
struct Bound final {};
struct Deferred final {};
} // namespace input
namespace detail {
template <class Mode>
concept InputMode =
    std::same_as<Mode, input::Bound> || std::same_as<Mode, input::Deferred>;
template <class... Values> struct InputSet final {};
template <class Tag, class Node> struct FieldNode final {
  Node value;
};
} // namespace detail
template <class Signature, class Stage = stage::Exact,
          class Mode = input::Bound>
class Flow;
namespace detail {
template <class Signature, class Stage = stage::Exact>
using DeferredFlow = Flow<Signature, Stage, input::Deferred>;
template <class Recipe>
  requires requires(Recipe &&recipe) { std::move(recipe).compile(); }
[[nodiscard]] auto compile_async(const std::shared_ptr<FlowState> &state,
                                 Recipe recipe);
} // namespace detail
template <class T, class Card = stage::Exact> class StageRef;
template <class... Stages> class ZipRef;
template <class Key, class Value, class Count = std::uint32_t> class Groups;
template <class T, class Count = CountFor<T>> class GroupValuesRef;
template <class... Schema> class RecordRef;
template <class... Schema> class Selection;
namespace detail {
struct NodeAccess;
template <class Schema> struct SchemaRef;

} // namespace detail
} // namespace rund::compute
