#pragma once

#include <rund/compute/pipeline/bind.hpp>
#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/program.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace rund::compute {

namespace detail {

template <class SeedSignature, class ActionSignature, class FoldSignature>
struct TileRepeatContract final {
  using SeedInputs = typename SignatureTypes<SeedSignature>::Inputs;
  using SeedOutputs = typename SignatureTypes<SeedSignature>::Outputs;
  using SeedCoordinates = WindowCoordinateSplit<SeedInputs>;
  using SeedExternalInputs = typename SeedCoordinates::Prefix;
  using ActionInputs = typename SignatureTypes<ActionSignature>::Inputs;
  using ActionOutputs = typename SignatureTypes<ActionSignature>::Outputs;
  using FoldInputs = typename SignatureTypes<FoldSignature>::Inputs;
  using FoldOutputs = typename SignatureTypes<FoldSignature>::Outputs;
  using ExpectedFoldInputs = typename Join<FoldOutputs, SeedOutputs>::type;

  static constexpr bool valid =
      SeedCoordinates::valid && std::is_same_v<ActionInputs, SeedOutputs> &&
      StartsWith<ActionOutputs, ActionInputs>::value &&
      std::is_same_v<FoldInputs, ExpectedFoldInputs>;
};

struct TileRepeatFactory;

} // namespace detail

template <std::size_t N, class SeedSignature, class ActionSignature,
          class FoldSignature>
class TileRepeat final {
public:
  TileRepeat(const TileRepeat &) noexcept = default;
  TileRepeat &operator=(const TileRepeat &) = delete;
  TileRepeat(TileRepeat &&) noexcept = default;
  TileRepeat &operator=(TileRepeat &&) = delete;

private:
  friend class PipelineBuilder;
  friend struct detail::TileRepeatFactory;

  TileRepeat(Program<SeedSignature> seed, Program<ActionSignature> action,
             Program<FoldSignature> fold) noexcept
      : seed_(std::move(seed)), action_(std::move(action)),
        fold_(std::move(fold)) {}

  const Program<SeedSignature> seed_;
  const Program<ActionSignature> action_;
  const Program<FoldSignature> fold_;
};

namespace detail {

struct TileRepeatFactory final {
  template <std::size_t N, class SeedSignature, class ActionSignature,
            class FoldSignature>
  [[nodiscard]] static TileRepeat<N, SeedSignature, ActionSignature,
                                  FoldSignature>
  make(const Program<SeedSignature> &seed,
       const Program<ActionSignature> &action,
       const Program<FoldSignature> &fold) noexcept {
    return TileRepeat<N, SeedSignature, ActionSignature, FoldSignature>{
        seed, action, fold};
  }
};

} // namespace detail

template <std::size_t N, class SeedSignature, class ActionSignature,
          class FoldSignature>
  requires(N != 0u && N <= PipelineInnerIterationCapacity &&
           detail::TileRepeatContract<SeedSignature, ActionSignature,
                                      FoldSignature>::valid)
[[nodiscard]] auto tile_repeat(const Program<SeedSignature> &seed,
                               const Program<ActionSignature> &action,
                               const Program<FoldSignature> &fold) noexcept {
  return detail::TileRepeatFactory::make<N>(seed, action, fold);
}

} // namespace rund::compute
