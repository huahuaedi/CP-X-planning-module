"""Apollo-style planning pipeline helpers.

This package keeps the high-level planning stages explicit:
cooperative perception input, prediction, behavior decision, and trajectory generation.
"""

from .candidate_evaluation import (
    BehaviorCandidate,
    CandidateEvaluationFrame,
    evaluate_behavior_candidates,
)
from .actuator_mapper import ActuatorCommand, CarlaActuatorMapper
from .architecture_profile import ArchitectureProfile, normalize_architecture_config
from .candidate_pipeline import (
    CandidateBehaviorIntent,
    CandidateReferenceResult,
    CandidateSelectionOutcome,
    build_candidate_intents,
    evaluate_candidate_reference,
    select_best_candidate,
    select_candidate_with_commitment,
    summarize_candidate_results,
)
from .control_buffer import MPCControlBuffer
from .behavior_decision import BehaviorConstraint, BehaviorDecision
from .behavior_stage import BehaviorStage, BehaviorStageResult
from .fallback_manager import (
    FailureReason,
    FallbackRequest,
    FallbackResult,
    TrajectoryFallbackManager,
)
from .decision_record import DecisionRecord, DecisionVeto, build_decision_record
from .destination_speed_stage import (
    DestinationSpeedStage,
    DestinationSpeedStageResult,
)
from .mpc_feedback import BehaviorMPCFeedback
from .mpc_command_extractor import MPCCommandExtractor, MPCTrackingCommand
from .mpc_entry_stage import MPCEntryStage, MPCEntryStageResult
from .maneuver_manager import LaneChangeLifecycle, ManeuverManager, TurnLifecycle
from .nominal_trajectory import (
    NominalTarget,
    NominalTrajectory,
    NominalTrajectoryGenerator,
)
from .prediction import PredictionFrame, build_prediction_frame
from .output import BehaviorCommand, PlannerDiagnostics, PlannerOutput
from .perception_stage import PerceptionStage, PerceptionStageResult
from .reference_contract import (
    ReferenceContract,
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)
from .reference_gate import FinalReferenceGate, FinalReferenceGateResult
from .reference_line_provider import (
    ReferenceLineProvider,
    ReferenceLineRequest,
    ReferenceLineResult,
)
from .reference_generator import (
    BoundaryRecoveryValidation,
    DrivableFootprintOccupancy,
    GeneratedReference,
    LaneCorridorOccupancy,
    ReferenceCorridorProjection,
    ReferenceGenerator,
)
from .reference_pipeline import (
    ConditionedReference,
    ReferencePipeline,
    ReferencePipelineRequest,
    ReferencePipelineResult,
)
from .reference_publication_stage import (
    ReferencePublicationStage,
    ReferencePublicationStageResult,
)
from .runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot
from .traffic_light_memory import TrafficLightMemory
from .stage_contracts import ManeuverCommitment
from .route_authorization import (
    LaneChangeAuthorization,
    RouteManeuver,
    authorize_route_lane_change,
    normalize_route_maneuver,
)
from .route_manager import CPXRouteManager, RouteCursorSnapshot, RouteManagerStatus
from .safety_supervisor import SafetySupervisor
from .scenario_manager import (
    BoundaryRecoveryRequest,
    CPXScenarioDecision,
    CPXScenarioManager,
)
from .speed_planner import (
    SpeedConstraint,
    SpeedPlan,
    SpeedTarget,
    SpeedTargetPlanner,
    build_speed_plan,
)
from .velocity_steering_adapter import (
    CarlaVelocitySteeringAdapter,
    OpenCDAVelocitySteeringAdapter,
    VelocitySteeringCommand,
)
from .tracker import CPXObstacleTracker

__all__ = [
    "ActuatorCommand",
    "BehaviorMPCFeedback",
    "MPCCommandExtractor",
    "MPCTrackingCommand",
    "MPCEntryStage",
    "MPCEntryStageResult",
    "CarlaActuatorMapper",
    "BoundaryRecoveryRequest",
    "BoundaryRecoveryValidation",
    "DrivableFootprintOccupancy",
    "ArchitectureProfile",
    "BehaviorCandidate",
    "BehaviorCommand",
    "CandidateEvaluationFrame",
    "CandidateBehaviorIntent",
    "CandidateReferenceResult",
    "CandidateSelectionOutcome",
    "CPXRouteManager",
    "RouteCursorSnapshot",
    "CPXObstacleTracker",
    "CPXScenarioDecision",
    "CPXScenarioManager",
    "DecisionRecord",
    "DecisionVeto",
    "DestinationSpeedStage",
    "DestinationSpeedStageResult",
    "MPCControlBuffer",
    "BehaviorConstraint",
    "BehaviorDecision",
    "BehaviorStage",
    "BehaviorStageResult",
    "ManeuverManager",
    "LaneChangeLifecycle",
    "TurnLifecycle",
    "NominalTarget",
    "NominalTrajectory",
    "NominalTrajectoryGenerator",
    "FallbackResult",
    "FailureReason",
    "FallbackRequest",
    "TrajectoryFallbackManager",
    "PlannerDiagnostics",
    "PlannerOutput",
    "PerceptionStage",
    "PerceptionStageResult",
    "PredictionFrame",
    "ReferenceContract",
    "FinalReferenceGate",
    "FinalReferenceGateResult",
    "ReferenceGenerator",
    "ReferenceLineProvider",
    "ReferenceLineRequest",
    "ReferenceLineResult",
    "GeneratedReference",
    "LaneCorridorOccupancy",
    "ReferenceCorridorProjection",
    "ReferencePipeline",
    "ReferencePipelineRequest",
    "ReferencePipelineResult",
    "ReferencePublicationStage",
    "ReferencePublicationStageResult",
    "RuntimeInputStage",
    "RuntimeTickSnapshot",
    "ConditionedReference",
    "TrafficLightMemory",
    "ManeuverCommitment",
    "ReferenceValidationResult",
    "LaneChangeAuthorization",
    "RouteManagerStatus",
    "RouteManeuver",
    "SafetySupervisor",
    "SpeedPlan",
    "SpeedConstraint",
    "SpeedTarget",
    "SpeedTargetPlanner",
    "CarlaVelocitySteeringAdapter",
    "OpenCDAVelocitySteeringAdapter",
    "VelocitySteeringCommand",
    "authorize_route_lane_change",
    "build_prediction_frame",
    "build_candidate_intents",
    "build_decision_record",
    "build_speed_plan",
    "contract_from_config",
    "evaluate_behavior_candidates",
    "evaluate_candidate_reference",
    "normalize_route_maneuver",
    "normalize_architecture_config",
    "select_best_candidate",
    "select_candidate_with_commitment",
    "summarize_candidate_results",
    "validate_reference_contract",
]
