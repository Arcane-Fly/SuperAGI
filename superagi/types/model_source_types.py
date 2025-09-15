from enum import Enum


class ModelSourceType(Enum):
    GooglePalm = 'Google Palm'
    OpenAI = 'OpenAi'
    Replicate = 'Replicate'
    HuggingFace = 'Hugging Face'
    LocalLLM = 'Local LLM'
    Anthropic = 'Anthropic'  # Added support for Anthropic Claude models

    @classmethod
    def get_model_source_type(cls, name):
        name = name.upper().replace(" ", "")
        for member in cls.__members__:
            if name == member.upper():
                return cls[member]
        raise ValueError(f"{name} is not a valid vector store name.")

    @classmethod
    def get_model_source_from_model(cls, model_name: str):
        # Updated OpenAI models list to include latest GPT-4 and GPT-3.5 variants as of 2024
        open_ai_models = [
            'gpt-4', 'gpt-4-32k', 'gpt-4-0125-preview', 'gpt-4-1106-preview',
            'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o', 'gpt-4o-mini',
            'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0125', 
            'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-instruct'
        ]
        # Updated Google models to include latest PaLM and Gemini models
        google_models = [
            'google-palm-bison-001', 'models/chat-bison-001',
            'gemini-pro', 'gemini-pro-vision', 'models/gemini-pro',
            'models/gemini-pro-vision'
        ]
        # Updated Replicate models 
        replicate_models = [
            'replicate-llama13b-v2-chat', 'llama-2-70b-chat',
            'llama-2-13b-chat', 'llama-2-7b-chat',
            'codellama-34b-instruct', 'codellama-13b-instruct'
        ]
        # Added Anthropic Claude models
        anthropic_models = [
            'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 
            'claude-3-haiku-20240307', 'claude-2.1', 'claude-2.0',
            'claude-instant-1.2'
        ]
        if model_name in open_ai_models:
            return ModelSourceType.OpenAI
        if model_name in google_models:
            return ModelSourceType.GooglePalm
        if model_name in replicate_models:
            return ModelSourceType.Replicate
        if model_name in anthropic_models:
            return ModelSourceType.Anthropic
        return ModelSourceType.OpenAI

    def __str__(self):
        return self.value