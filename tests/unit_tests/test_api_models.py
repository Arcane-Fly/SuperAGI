"""
Tests for SuperAGI API models and new provider integrations.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from superagi.types.api_models import (
    TaskStatus, TaskType, ProviderType, PersonaType, ModelCapability,
    ModelSpec, AIModelConfig, ExecutionContext, TaskRequest, TaskResponse,
    get_models_by_provider, get_model_by_name, get_latest_models
)


class TestEnums:
    """Test enum definitions."""
    
    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
    
    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.CODE_GENERATION == "code_generation"
        assert TaskType.CODE_ANALYSIS == "code_analysis"
        assert TaskType.SECURITY_AUDIT == "security_audit"
        assert TaskType.CUSTOM == "custom"
    
    def test_provider_type_enum(self):
        """Test ProviderType enum includes new providers."""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.ANTHROPIC == "anthropic"
        assert ProviderType.GOOGLE == "google"
        assert ProviderType.GROK == "grok"
        assert ProviderType.GROQ == "groq"
        assert ProviderType.REPLICATE == "replicate"
    
    def test_persona_type_enum(self):
        """Test PersonaType enum values."""
        assert PersonaType.DEVELOPER == "developer"
        assert PersonaType.SECURITY_ANALYST == "security_analyst"
        assert PersonaType.PERFORMANCE_EXPERT == "performance_expert"
        assert PersonaType.CUSTOM == "custom"


class TestModelSpec:
    """Test ModelSpec model."""
    
    def test_valid_model_spec(self):
        """Test creating a valid ModelSpec."""
        spec = ModelSpec(
            name="gpt-4o",
            provider=ProviderType.OPENAI,
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
            context_length=128000,
            max_output_tokens=16384,
            input_cost_per_token=0.0000025,
            output_cost_per_token=0.00001,
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
            release_date="2024-05-13",
            description="GPT-4o: Advanced multimodal model"
        )
        
        assert spec.name == "gpt-4o"
        assert spec.provider == ProviderType.OPENAI
        assert spec.context_length == 128000
        assert spec.supports_vision is True
    
    def test_model_spec_validation(self):
        """Test ModelSpec validation."""
        with pytest.raises(ValidationError):
            ModelSpec(
                name="",  # Empty name should fail
                provider=ProviderType.OPENAI,
                capabilities=[],
                context_length=128000,
                description="Test model"
            )


class TestAIModelConfig:
    """Test AIModelConfig model."""
    
    def test_valid_model_config(self):
        """Test creating a valid AIModelConfig."""
        config = AIModelConfig(
            model="gpt-4o",
            provider=ProviderType.OPENAI,
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9
        )
        
        assert config.model == "gpt-4o"
        assert config.provider == ProviderType.OPENAI
        assert config.temperature == 0.7
    
    def test_model_config_validation(self):
        """Test AIModelConfig validation."""
        # Test empty model name
        with pytest.raises(ValidationError):
            AIModelConfig(
                model="",
                provider=ProviderType.OPENAI
            )
        
        # Test invalid temperature
        with pytest.raises(ValidationError):
            AIModelConfig(
                model="gpt-4o",
                provider=ProviderType.OPENAI,
                temperature=3.0  # Should be <= 2.0
            )


class TestExecutionContext:
    """Test ExecutionContext model."""
    
    def test_valid_execution_context(self):
        """Test creating a valid ExecutionContext."""
        context = ExecutionContext(
            user_id="user123",
            session_id="session456",
            workspace_id="workspace789",
            timeout=600,
            priority=5
        )
        
        assert context.user_id == "user123"
        assert context.timeout == 600
        assert context.priority == 5
    
    def test_execution_context_defaults(self):
        """Test ExecutionContext default values."""
        context = ExecutionContext(user_id="user123")
        
        assert context.environment == "development"
        assert context.timeout == 300
        assert context.priority == 1


class TestTaskRequest:
    """Test TaskRequest model."""
    
    def test_valid_task_request(self):
        """Test creating a valid TaskRequest."""
        model_config = AIModelConfig(
            model="gpt-4o",
            provider=ProviderType.OPENAI
        )
        
        context = ExecutionContext(user_id="user123")
        
        task = TaskRequest(
            type=TaskType.CODE_GENERATION,
            title="Generate API endpoints",
            description="Create FastAPI endpoints",
            context=context,
            model_config=model_config
        )
        
        assert task.type == TaskType.CODE_GENERATION
        assert task.persona == PersonaType.DEVELOPER  # Default
        assert isinstance(task.created_at, datetime)
        assert len(task.id) > 0  # UUID should be generated


class TestTaskResponse:
    """Test TaskResponse model."""
    
    def test_valid_task_response(self):
        """Test creating a valid TaskResponse."""
        response = TaskResponse(
            id="task123",
            status=TaskStatus.COMPLETED,
            result={"output": "Generated code"},
            execution_time=45.2,
            tokens_used=1500,
            cost=0.03,
            created_at=datetime.utcnow()
        )
        
        assert response.id == "task123"
        assert response.status == TaskStatus.COMPLETED
        assert response.execution_time == 45.2


class TestModelRegistry:
    """Test model registry functions."""
    
    def test_get_models_by_provider(self):
        """Test getting models by provider."""
        openai_models = get_models_by_provider(ProviderType.OPENAI)
        anthropic_models = get_models_by_provider(ProviderType.ANTHROPIC)
        groq_models = get_models_by_provider(ProviderType.GROQ)
        
        assert len(openai_models) > 0
        assert len(anthropic_models) > 0
        assert len(groq_models) > 0
        
        # Check that all returned models have correct provider
        for model in openai_models:
            assert model.provider == ProviderType.OPENAI
    
    def test_get_model_by_name(self):
        """Test getting specific model by name."""
        gpt4o = get_model_by_name("gpt-4o")
        claude = get_model_by_name("claude-3-5-sonnet-20241022")
        groq_llama = get_model_by_name("llama-3.1-70b-versatile")
        
        assert gpt4o is not None
        assert gpt4o.provider == ProviderType.OPENAI
        assert gpt4o.supports_vision is True
        
        assert claude is not None
        assert claude.provider == ProviderType.ANTHROPIC
        
        assert groq_llama is not None
        assert groq_llama.provider == ProviderType.GROQ
        
        # Test non-existent model
        assert get_model_by_name("non-existent-model") is None
    
    def test_get_latest_models(self):
        """Test getting latest 2024 models."""
        latest_models = get_latest_models()
        
        assert len(latest_models) > 0
        
        # All models should be from 2024
        for model in latest_models:
            assert model.release_date is not None
            assert model.release_date.startswith("2024")
    
    def test_model_capabilities(self):
        """Test model capabilities are correctly assigned."""
        # Test GPT-4o capabilities
        gpt4o = get_model_by_name("gpt-4o")
        assert ModelCapability.TEXT_GENERATION in gpt4o.capabilities
        assert ModelCapability.CODE_GENERATION in gpt4o.capabilities
        assert ModelCapability.VISION in gpt4o.capabilities
        assert ModelCapability.FUNCTION_CALLING in gpt4o.capabilities
        
        # Test o1-preview capabilities (no vision or function calling)
        o1_preview = get_model_by_name("o1-preview")
        assert ModelCapability.REASONING in o1_preview.capabilities
        assert ModelCapability.VISION not in o1_preview.capabilities
        assert ModelCapability.FUNCTION_CALLING not in o1_preview.capabilities
        
        # Test Groq model capabilities
        groq_model = get_model_by_name("llama-3.1-70b-versatile")
        assert ModelCapability.TEXT_GENERATION in groq_model.capabilities
        assert ModelCapability.CODE_GENERATION in groq_model.capabilities
        assert groq_model.supports_vision is False  # Groq models don't support vision
    
    def test_model_costs(self):
        """Test model cost information."""
        gpt4o_mini = get_model_by_name("gpt-4o-mini")
        o1_preview = get_model_by_name("o1-preview")
        groq_model = get_model_by_name("llama-3.1-8b-instant")
        
        # GPT-4o-mini should be cheaper than o1-preview
        assert gpt4o_mini.input_cost_per_token < o1_preview.input_cost_per_token
        assert gpt4o_mini.output_cost_per_token < o1_preview.output_cost_per_token
        
        # Groq should be very cost-effective
        assert groq_model.input_cost_per_token < gpt4o_mini.input_cost_per_token
    
    def test_context_lengths(self):
        """Test model context lengths."""
        gemini_pro = get_model_by_name("gemini-1.5-pro")
        gpt4o = get_model_by_name("gpt-4o")
        claude = get_model_by_name("claude-3-5-sonnet-20241022")
        
        # Gemini 1.5 Pro should have the largest context
        assert gemini_pro.context_length == 2000000  # 2M tokens
        
        # GPT-4o should have 128K context
        assert gpt4o.context_length == 128000
        
        # Claude should have 200K context
        assert claude.context_length == 200000


if __name__ == "__main__":
    pytest.main([__file__])