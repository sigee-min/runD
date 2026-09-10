#include "run.hpp"

#include "backend.hpp"
#include "cache.hpp"
#include "capacity.hpp"
#include "claim.hpp"
#include "failure.hpp"
#include "manifest.hpp"
#include "map.hpp"
#include "numeric.hpp"
#include "phase.hpp"
#include "publication.hpp"
#include "recipe.hpp"
#include "route.hpp"
#include "window.hpp"

#include <cstdio>

namespace node_accel_contract {

bool AuthorityContract() {
#define CHECK_AUTHORITY(name)                                                  \
  do {                                                                         \
    if (!name()) {                                                             \
      std::fprintf(stderr, "authority contract failed: %s\n", #name);          \
      return false;                                                            \
    }                                                                          \
  } while (false)
  CHECK_AUTHORITY(BackendWindowDefaultsFailClosed);
  CHECK_AUTHORITY(BackendWindowOccurrenceShapeHasOneAuthority);
  CHECK_AUTHORITY(SubmissionTransitions);
  CHECK_AUTHORITY(PreparedPipelineClaimHasOneAuthority);
  CHECK_AUTHORITY(PhaseSourceIdentityContract);
  CHECK_AUTHORITY(PreparedPipelineFailureCoordinatesAreExact);
  CHECK_AUTHORITY(PreparedPipelinePreparePathsAlwaysReportFailure);
  CHECK_AUTHORITY(PreparedTemplateRegistryIsColdAndCollisionSafe);
  CHECK_AUTHORITY(PreparedReservationIsFieldwiseFailClosed);
  CHECK_AUTHORITY(RecurrenceRouteCopiesDoNotCloneTemplates);
  CHECK_AUTHORITY(MetalPointerIdentityIndexIsExactAndOneShot);
  CHECK_AUTHORITY(MetalIcbSizeClassPlanIsExactAndPortable);
  CHECK_AUTHORITY(MetalColdIdentityIndexBudgetIsExact);
  CHECK_AUTHORITY(MetalCaptureRowCapacityIsExact);
  CHECK_AUTHORITY(PreparedTemplateStepCapacityIsBackendBounded);
  CHECK_AUTHORITY(PreparedBackendProjectionIsRouteWiseAndChecked);
  CHECK_AUTHORITY(PreparedPublicationCommandShapeIsCanonical);
  CHECK_AUTHORITY(VulkanPhysicalCommandShapeIsNonOverlapping);
  CHECK_AUTHORITY(VulkanPhysicalCapacityFailureIsTransactional);
  CHECK_AUTHORITY(PreparedBackendControlManifestIsDimensionallyClosed);
  CHECK_AUTHORITY(BackendSourceRecipeIsCheckedAndCanonical);
  CHECK_AUTHORITY(VulkanMapAndResetSourceRecipesAreExact);
  CHECK_AUTHORITY(MetalMapWordClassPartitionsProgramTemplates);
  CHECK_AUTHORITY(MapSourceSpecializationIsSingleOwnerAndExact);
  CHECK_AUTHORITY(MapArithmeticMeaningIsCanonical);
  CHECK_AUTHORITY(MetalSourceRecipesAreExactAndSemantic);
  CHECK_AUTHORITY(MetalMapCheckSourceHasOneGuardAuthority);
  CHECK_AUTHORITY(MetalColdManifestIsExact);
  CHECK_AUTHORITY(BackendTemplateRouteDemandIsExactAndFrozen);
  CHECK_AUTHORITY(PreparedPublicationFingerprintIsSemantic);
  CHECK_AUTHORITY(PreparedPublicationResolvedIdentityIsExact);
  CHECK_AUTHORITY(GridBoundaries);
  CHECK_AUTHORITY(FinishPrecedence);
  CHECK_AUTHORITY(TelemetryProjection);
  CHECK_AUTHORITY(BackendCapacityExceptionClassesAreCanonical);
#undef CHECK_AUTHORITY
  return true;
}

} // namespace node_accel_contract
