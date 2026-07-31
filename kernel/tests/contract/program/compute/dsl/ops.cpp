#include "contract/program/compute/dsl/ops/local.hpp"

namespace program_compute_contract {

int RunComputeDslOpsContract() {
  for (auto run : {
      RunComputeDslScalarOpsContract, RunComputeDslBitOpsContract,
      RunComputeDslOpsHashContract,   RunComputeDslOpsNoiseContract,
      RunComputeDslOpsNoiseGridContract, RunComputeDslOpsNormContract,
      RunComputeDslOpsVecContract,    RunComputeDslOpsSqContract,
      RunComputeDslOpsMetricContract, RunComputeDslOpsRangeContract,
      RunComputeDslOpsAggregateContract, RunComputeDslOpsStatsContract,
      RunComputeDslOpsMomentContract, RunComputeDslOpsCorrContract,
      RunComputeDslOpsRatioContract,  RunComputeDslOpsStandardizeContract,
      RunComputeDslOpsApproxContract,
      RunComputeDslOpsTranscendentalContract,
      RunComputeDslOpsComplexContract,
      RunComputeDslOpsWindowSignalContract,
      RunComputeDslOpsLinearContract,
      RunComputeDslOpsMatrixContract, RunComputeDslOpsAffineContract,
      RunComputeDslOpsMixContract,    RunComputeDslOpsPolyContract,
      RunComputeDslOpsInterpContract, RunComputeDslOpsMaskContract,
      RunComputeDslOpsTolContract,    RunComputeDslOpsPieceContract,
      RunComputeDslOpsProjContract,   RunComputeDslOpsReflectContract,
      RunComputeDslOpsCrossContract,  RunComputeDslOpsLineContract,
      RunComputeDslOpsPlaneContract,  RunComputeDslNonlinearOpsContract,
      RunComputeDslOpsOverloadContract,
  }) {
    if (run() != 0) {
      return 1;
    }
  }
  return 0;
}

} // namespace program_compute_contract
