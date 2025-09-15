"""
Pydantic models for SuperAGI Core API.

This module defines all the data models used for API requests and responses,
including validation, serialization, and documentation with support for
the latest AI models from major providers (OpenAI, Anthropic, Google, Grok, Groq).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Task type enumeration for AI-powered development tasks."""
    
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CUSTOM = "custom"


class ProviderType(str, Enum):
    """AI provider type enumeration with latest supported providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROK = "grok"  # xAI's Grok models
    GROQ = "groq"  # Hardware-accelerated inference provider
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    LOCAL_LLM = "local_llm"


class PersonaType(str, Enum):
    """AI persona types for specialized assistance."""
    
    DEVELOPER = "developer"
    ARCHITECT = "architect"
    REVIEWER = "reviewer"
    SECURITY_ANALYST = "security_analyst"
    PERFORMANCE_EXPERT = "performance_expert"
    TESTER = "tester"
    TECHNICAL_WRITER = "technical_writer"
    DEVOPS_ENGINEER = "devops_engineer"
    DATA_SCIENTIST = "data_scientist"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """Model capability enumeration."""
    
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"


class ModelSpec(BaseModel):
    """Specification for an AI model."""
    
    name: str = Field(..., description="Model name/identifier")
    provider: ProviderType = Field(..., description="Model provider")
    capabilities: List[ModelCapability] = Field(..., description="Model capabilities")
    context_length: int = Field(..., description="Maximum context length in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    input_cost_per_token: Optional[float] = Field(None, description="Input cost per token (USD)")
    output_cost_per_token: Optional[float] = Field(None, description="Output cost per token (USD)")
    supports_streaming: bool = Field(default=True, description="Supports streaming responses")
    supports_functions: bool = Field(default=False, description="Supports function calling")
    supports_vision: bool = Field(default=False, description="Supports vision/image inputs")
    release_date: Optional[str] = Field(None, description="Model release date (YYYY-MM-DD)")
    deprecation_date: Optional[str] = Field(None, description="Model deprecation date")
    description: str = Field(..., description="Model description")


class AIModelConfig(BaseModel):
    """Configuration for AI model usage."""
    
    model: str = Field(..., description="Model identifier")
    provider: ProviderType = Field(..., description="Model provider")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    system_prompt: Optional[str] = Field(None, description="System prompt")


class ExecutionContext(BaseModel):
    """Execution context for task processing."""
    
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier")
    project_id: Optional[str] = Field(None, description="Project identifier")
    environment: str = Field(default="development", description="Execution environment")
    timeout: int = Field(default=300, description="Task timeout in seconds")
    priority: int = Field(default=1, description="Task priority (1-10)")


class TaskRequest(BaseModel):
    """Request model for task creation."""
    
    task_id: str
    task_type: TaskType
    title: str
    description: str
    user_id: str
    ai_model: str  # Changed from model_name to avoid pydantic conflict
    provider: ProviderType
    persona: PersonaType = PersonaType.DEVELOPER
    parameters: dict = {}


class TaskResponse(BaseModel):
    """Response model for task results."""
    
    task_id: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


# Latest AI Models as of 2024 - Updated with newest releases

OPENAI_MODELS = [
    ModelSpec(
        name="gpt-4o",
        provider=ProviderType.OPENAI,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.VISION, ModelCapability.FUNCTION_CALLING],
        context_length=128000,
        max_output_tokens=16384,
        input_cost_per_token=0.0000025,
        output_cost_per_token=0.00001,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=True,
        release_date="2024-05-13",
        description="GPT-4o: Advanced multimodal model with vision and reasoning capabilities"
    ),
    ModelSpec(
        name="gpt-4o-2024-08-06",
        provider=ProviderType.OPENAI,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.VISION, ModelCapability.FUNCTION_CALLING],
        context_length=128000,
        max_output_tokens=16384,
        input_cost_per_token=0.0000025,
        output_cost_per_token=0.00001,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=True,
        release_date="2024-08-06",
        description="GPT-4o August 2024: Latest version with improved performance and capabilities"
    ),
    ModelSpec(
        name="gpt-4o-mini",
        provider=ProviderType.OPENAI,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_length=128000,
        max_output_tokens=16384,
        input_cost_per_token=0.00000015,
        output_cost_per_token=0.0000006,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=False,
        release_date="2024-07-18",
        description="GPT-4o-mini: Cost-effective model with strong performance for most tasks"
    ),
    ModelSpec(
        name="o1-preview",
        provider=ProviderType.OPENAI,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=128000,
        max_output_tokens=32768,
        input_cost_per_token=0.000015,
        output_cost_per_token=0.00006,
        supports_streaming=False,
        supports_functions=False,
        supports_vision=False,
        release_date="2024-09-12",
        description="o1-preview: Advanced reasoning model for complex problem-solving"
    ),
    ModelSpec(
        name="o1-mini",
        provider=ProviderType.OPENAI,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=128000,
        max_output_tokens=65536,
        input_cost_per_token=0.000003,
        output_cost_per_token=0.000012,
        supports_streaming=False,
        supports_functions=False,
        supports_vision=False,
        release_date="2024-09-12",
        description="o1-mini: Faster reasoning model optimized for STEM tasks"
    ),
]

ANTHROPIC_MODELS = [
    ModelSpec(
        name="claude-3-5-sonnet-20241022",
        provider=ProviderType.ANTHROPIC,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.VISION, ModelCapability.FUNCTION_CALLING],
        context_length=200000,
        max_output_tokens=8192,
        input_cost_per_token=0.000003,
        output_cost_per_token=0.000015,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=True,
        release_date="2024-10-22",
        description="Claude 3.5 Sonnet: Latest model with enhanced reasoning and coding capabilities"
    ),
    ModelSpec(
        name="claude-3-5-haiku-20241022",
        provider=ProviderType.ANTHROPIC,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=200000,
        max_output_tokens=8192,
        input_cost_per_token=0.0000008,
        output_cost_per_token=0.000004,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=False,
        release_date="2024-10-22",
        description="Claude 3.5 Haiku: Fast and efficient model for quick tasks"
    ),
]

GOOGLE_MODELS = [
    ModelSpec(
        name="gemini-1.5-pro",
        provider=ProviderType.GOOGLE,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.VISION, ModelCapability.MULTIMODAL],
        context_length=2000000,
        max_output_tokens=8192,
        input_cost_per_token=0.00000125,
        output_cost_per_token=0.000005,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=True,
        release_date="2024-05-14",
        description="Gemini 1.5 Pro: Most capable Google model with massive context window"
    ),
    ModelSpec(
        name="gemini-1.5-flash",
        provider=ProviderType.GOOGLE,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                     ModelCapability.REASONING, ModelCapability.VISION, ModelCapability.MULTIMODAL],
        context_length=1000000,
        max_output_tokens=8192,
        input_cost_per_token=0.000000075,
        output_cost_per_token=0.0000003,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=True,
        release_date="2024-05-14",
        description="Gemini 1.5 Flash: Fast and efficient model with large context"
    ),
]

GROK_MODELS = [
    ModelSpec(
        name="grok-beta",
        provider=ProviderType.GROK,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=131072,
        max_output_tokens=4096,
        input_cost_per_token=0.000005,
        output_cost_per_token=0.000015,
        supports_streaming=True,
        supports_functions=False,
        supports_vision=False,
        release_date="2024-03-17",
        description="Grok: xAI's conversational AI with real-time knowledge and humor"
    ),
]

GROQ_MODELS = [
    ModelSpec(
        name="llama-3.1-70b-versatile",
        provider=ProviderType.GROQ,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=131072,
        max_output_tokens=32768,
        input_cost_per_token=0.00000059,
        output_cost_per_token=0.00000079,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=False,
        release_date="2024-07-23",
        description="Llama 3.1 70B on Groq: Ultra-fast inference with Meta's latest model"
    ),
    ModelSpec(
        name="llama-3.1-8b-instant",
        provider=ProviderType.GROQ,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.REASONING],
        context_length=131072,
        max_output_tokens=32768,
        input_cost_per_token=0.00000005,
        output_cost_per_token=0.00000008,
        supports_streaming=True,
        supports_functions=True,
        supports_vision=False,
        release_date="2024-07-23",
        description="Llama 3.1 8B on Groq: Lightning-fast inference for quick tasks"
    ),
]

# Consolidated model registry
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS + GOOGLE_MODELS + GROK_MODELS + GROQ_MODELS

def get_models_by_provider(provider: ProviderType) -> List[ModelSpec]:
    """Get all models for a specific provider."""
    return [model for model in ALL_MODELS if model.provider == provider]

def get_model_by_name(name: str) -> Optional[ModelSpec]:
    """Get a specific model by name."""
    return next((model for model in ALL_MODELS if model.name == name), None)

def get_latest_models() -> List[ModelSpec]:
    """Get the latest models (released in 2024)."""
    return [model for model in ALL_MODELS if model.release_date and model.release_date.startswith("2024")]