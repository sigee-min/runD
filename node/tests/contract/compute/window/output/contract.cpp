#include "../../allocation.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
[[nodiscard]] int CheckWindowOutput(Device &device, const Backend backend,
                                    WindowOutputIdentity &identity) {
  auto seed = MakeOutputSeedProgram(device, false);
  auto seed_fault = MakeOutputSeedProgram(device, true);
  auto action = MakeOutputActionProgram(device, false);
  auto action_fault = MakeOutputActionProgram(device, true);
  auto fold = MakeOutputFoldProgram(device);
  auto high_index_fold = MakeOutputHighIndexFoldProgram(device);
  auto fold_two = MakeOutputFoldTwoProgram(device);
  auto priority_fold = MakeOutputPriorityFoldProgram(device);
  auto scatter_seed = MakeOutputScatterSeedProgram(device);
  auto scatter_fold = MakeOutputScatterFoldProgram(device);
  auto downstream = MakeOutputDownstreamProgram(device);
  auto count_advance = MakeOutputCountAdvanceProgram(device);
  auto zero_seed = MakeOutputZeroSeedProgram(device);
  auto zero_fold = MakeOutputZeroFoldProgram(device);
  if (!seed || !seed_fault || !action || !action_fault || !fold ||
      !high_index_fold || !fold_two || !priority_fold || !scatter_seed ||
      !scatter_fold || !downstream || !count_advance || !zero_seed ||
      !zero_fold) {
    return 1;
  }
  if (!PublicationFingerprintV3Golden()) {
    return 2;
  }
  if (!PublicationSourceCoordinates()) {
    return 3;
  }
  if (const int rollback =
          CheckOutputBuildAllocationRollback(device, *seed, *fold);
      rollback != 0) {
    return 160 + rollback;
  }
  if (const int aliases = CheckOrdinaryAliasAuthority(device); aliases != 0) {
    return 140 + aliases;
  }
  if (const int subview = CheckAliasSubviewRouting(device); subview != 0) {
    return 150 + subview;
  }
  if (const int mutation =
          CheckOutputSealedPublicationMutation(device, *seed, *fold);
      mutation != 0) {
    return 3 + mutation;
  }
  if (const int mutation =
          CheckOutputPublicationJobBindingMutation(device, *seed, *fold);
      mutation != 0) {
    return 40 + mutation;
  }
  if (const int parity = CheckOutputTransactionalCountParity(
          device, *seed, *fold, *count_advance);
      parity != 0) {
    return 120 + parity;
  }
  if (const int target =
          CheckOutputStatePairPublicationTargetRejected(device, *seed, *fold);
      target != 0) {
    return 130 + target;
  }
  for (const std::uint32_t count : kOutputCounts) {
    const int checked = CheckOutputCount(device, backend, *seed, *fold, count,
                                         kOutputValues, false, identity);
    if (checked != 0) {
      return 10 + checked;
    }
  }
  if (const int high_index =
          CheckOutputCount(device, backend, *seed, *high_index_fold,
                           kOutputMaximum, kOutputValues, true, identity);
      high_index != 0) {
    return 20 + high_index;
  }
  if (const int scatter = CheckOutputScatterConflicts(
          device, backend, *scatter_seed, *scatter_fold, identity);
      scatter != 0) {
    return 30 + scatter;
  }
  if (const int downstream_read = CheckOutputDownstreamRead(
          device, backend, *seed, *fold, *downstream, identity);
      downstream_read != 0) {
    return 40 + downstream_read;
  }
  if (const int zero_prefix = CheckOutputZeroRecurrentPrefix(
          device, backend, *zero_seed, *zero_fold, identity);
      zero_prefix != 0) {
    return 60 + zero_prefix;
  }
  if (const int arity = CheckOutputPublicationArity(device, *seed, *action);
      arity != 0) {
    return 70 + arity;
  }
  if (const int failure =
          CheckOutputLateFailures(device, backend, *seed, *seed_fault,
                                  *action_fault, *fold, *priority_fold);
      failure != 0) {
    return failure;
  }
  const int aliases = CheckOutputAliases(device, *seed, *fold, *fold_two);
  return aliases == 0 ? 0 : 110 + aliases;
}

} // namespace rund::node::test_contract::window
