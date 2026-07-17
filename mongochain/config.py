"""Configuration classes for mongochain."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for a MongoAgent.

    Attributes:
        name: Agent name (becomes the MongoDB database name)
        persona: The agent's personality/system prompt
        mongo_uri: MongoDB Atlas connection string
        llm_api_key: API key for the chosen LLM provider
        llm_provider: LLM provider ("openai", "anthropic", "google", "azure_openai", "grove", or "ollama")
        llm_model: Specific model to use (uses provider default if None)
        collaborators: List of agent names this agent can read memories from
        azure_endpoint: Azure OpenAI endpoint URL (required for azure_openai provider)
        azure_api_version: Azure OpenAI API version (default: "2024-02-15-preview")
        embedding_model: Voyage AI model used by Atlas autoEmbed (default: "voyage-4")
    """
    name: str
    persona: str
    mongo_uri: str
    llm_api_key: str
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    collaborators: list[str] = field(default_factory=list)
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    embedding_model: str = "voyage-4"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_providers = {"openai", "anthropic", "google", "azure_openai", "grove", "ollama"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"Invalid llm_provider '{self.llm_provider}'. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        # Azure OpenAI requires endpoint
        if self.llm_provider == "azure_openai" and not self.azure_endpoint:
            raise ValueError(
                "azure_endpoint is required when using 'azure_openai' provider. "
                "Example: 'https://your-resource.openai.azure.com/'"
            )

        if not self.name:
            raise ValueError("Agent name cannot be empty")

        if not self.mongo_uri:
            raise ValueError("MongoDB URI cannot be empty")

        # Sanitize agent name for use as database name
        self.db_name = self._sanitize_db_name(self.name)

    @staticmethod
    def _sanitize_db_name(name: str) -> str:
        """Sanitize agent name for use as MongoDB database name.

        MongoDB database names cannot contain: /\\. "$*<>:|?
        and should be lowercase for consistency.
        """
        invalid_chars = '/\\. "$*<>:|?'
        sanitized = name.lower()
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")
        return sanitized


@dataclass
class MemoryConfig:
    """Configuration for memory storage.

    Attributes:
        conversation_ttl_days: TTL for conversation history (default: 90 days)
        short_term_ttl_days: TTL for short-term summaries (default: 7 days)
        short_term_limit: Max short-term summaries to retrieve
        long_term_limit: Max long-term memories to retrieve via vector search
        conversation_limit: Max conversation messages to include in context
        embedding_model: Voyage AI model used by Atlas autoEmbed
    """
    conversation_ttl_days: int = 90
    short_term_ttl_days: int = 7
    short_term_limit: int = 5
    long_term_limit: int = 5
    conversation_limit: int = 20
    embedding_model: str = "voyage-4"
