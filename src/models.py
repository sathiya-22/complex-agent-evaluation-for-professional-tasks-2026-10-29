from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class AgentStep(BaseModel):
    step_id: str
    thought: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    observation: Optional[str] = None
    output: Optional[str] = None # Final output of the step if it concludes a sub-task

class AgentTranscript(BaseModel):
    task_id: str
    steps: List[AgentStep]
    final_answer: str

class ExpectedToolCall(BaseModel):
    tool_name: str
    # Arguments can be partially specified or have regex patterns for validation
    # For simplicity, we'll start with exact match or "any" for value
    arguments: Dict[str, Any]

class ExpectedOutput(BaseModel):
    # For professional tasks, output might be a structured document, a calculation, etc.
    # We'll use a simple dictionary for now, expecting specific keys/values.
    expected_value: Any
    tolerance: Optional[float] = None # For numerical values

class TaskDefinition(BaseModel):
    task_name: str
    description: str
    expected_steps_overview: List[str] # High-level description of expected agent flow
    expected_tool_calls: List[ExpectedToolCall] = Field(default_factory=list)
    expected_final_output: ExpectedOutput

class GroundTruth(BaseModel):
    task_id: str
    expected_tool_calls_sequence: List[ExpectedToolCall]
    expected_final_answer: Any

class EvaluationResult(BaseModel):
    metric: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)

class OverallEvaluation(BaseModel):
    task_id: str
    overall_score: float
    results: List[EvaluationResult]
