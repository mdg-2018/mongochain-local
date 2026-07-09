# Mongochain

A simple Python agent framework that uses **MongoDB Atlas** as the memory layer, demonstrating how MongoDB can power stateful AI agents with user-specific memories.

Embeddings are handled by **MongoDB Atlas Vector Search Automated Embedding** (`autoEmbed`) — Atlas generates and manages Voyage AI embeddings server-side at index-time and query-time. You don't need a Voyage AI API key.

## Features

- **User-Specific Memory**: All memories tied to user email/ID
- **Three Memory Types**:
  - `conversation_history` — Raw chat log (90-day TTL)
  - `short_term_memory` — Session summaries (7-day TTL)
  - `long_term_memory` — Persistent user facts (no TTL)
- **Automated Embedding**: Atlas generates and manages vector embeddings — no client-side embedding code or API key
- **Auto-Extraction**: LLM automatically extracts and stores user facts
- **Dynamic Personas**: Change agent personality on the fly
- **Multi-Provider LLM**: Works with OpenAI, Azure OpenAI, Anthropic, Google, and Grove
- **Multi-Agent Collaboration**: Agents can share user memories

## Installation

Use a virtual environment. This avoids conflicts with system-managed Python installations (e.g., Homebrew, apt), which can fail during dependency upgrades with errors like *"Cannot uninstall typing_extensions ... no RECORD file"*.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/robin-mongodb/mongochain.git
```

Or install from source:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Running in Google Colab? Skip the venv — Colab already provides an isolated environment. Just `%pip install git+https://github.com/robin-mongodb/mongochain.git` in a cell.

## Quick Start

```python
from mongochain import MongoAgent

# Create an agent
agent = MongoAgent(
    name="assistant",
    persona="You are a helpful research assistant.",
    mongo_uri="mongodb+srv://...",
    llm_api_key="sk-..."
)

# Chat requires a user_id (email)
response = agent.chat(
    user_id="user@example.com",
    message="Hi! I'm a DevOps engineer working on MongoDB optimization."
)
print(response)

# The agent automatically extracts and stores user facts!
memories = agent.get_user_memories("user@example.com")
for m in memories:
    print(f"- {m['content']}")
```

## Memory Architecture

Each agent creates a MongoDB database with these collections. Vector search uses Atlas Automated Embedding on the text field noted below.

| Collection             | Purpose               | TTL     | Auto-embedded field |
| ---------------------- | --------------------- | ------- | ------------------- |
| `conversation_history` | Raw chat messages     | 90 days | `content`           |
| `short_term_memory`    | Session summaries     | 7 days  | `summary`           |
| `long_term_memory`     | Persistent user facts | None    | `content`           |

All memories are **user-specific** (filtered by `user_id`).

## Atlas Requirements for Automated Embedding

- **M0 (Free) or Flex tier**: works out of the box.
- **Dedicated (M10+)**: requires **Compute Auto-Scale** enabled (Atlas UI → Edit Configuration → Auto-scale cluster tier). Storage auto-scale is also recommended.
  - Current M10/M20 → set max instance size to M30 or higher.
  - Current M30+ → set max at least one tier higher.

If auto-scale is not enabled on a dedicated cluster, agent creation prints a clear failure line naming which vector index couldn't be created and why.

## Choosing an Embedding Model

Default is `voyage-4`. Override via the `embedding_model` kwarg:

```python
agent = MongoAgent(
    name="assistant",
    persona="...",
    mongo_uri=MONGO_URI,
    llm_api_key=LLM_KEY,
    embedding_model="voyage-4-large",  # or "voyage-4-lite", "voyage-code-3"
)
```

## Storing User Memories

### Automatic (via LLM)

The agent extracts significant facts from conversations:

```python
agent.chat("user@example.com", "I'm a senior engineer at Acme Corp")
```

### Manual

```python
agent.store_user_memory(
    user_id="user@example.com",
    content="User prefers detailed technical explanations",
    category="preference",
)
```

## Dynamic Personas

```python
agent.set_persona("You are a grumpy professor who gives brief answers.")
agent.chat("user@example.com", "Explain indexing")

agent.set_persona("You are a pirate who explains things with nautical metaphors.")
agent.chat("user@example.com", "Explain indexing")  # Different style!
```

## Multi-Agent Collaboration

```python
alice = MongoAgent(
    name="alice",
    persona="You are a research specialist.",
    mongo_uri=MONGO_URI,
    llm_api_key=LLM_KEY,
)

# Bob can read Alice's user memories
bob = MongoAgent(
    name="bob",
    persona="You are a summarization expert.",
    mongo_uri=MONGO_URI,
    llm_api_key=LLM_KEY,
    collaborators=["alice"],
)

response = bob.chat("user@example.com", "What has Alice learned about me?")
```

## Using Different LLM Providers

```python
# OpenAI (default)
agent = MongoAgent(name="assistant", llm_provider="openai", llm_api_key="sk-...", ...)

# Anthropic Claude
agent = MongoAgent(name="assistant", llm_provider="anthropic", llm_api_key="sk-ant-...", ...)

# Google Gemini
agent = MongoAgent(
    name="assistant",
    llm_provider="google",
    llm_api_key="AIza...",
    llm_model="gemini-1.5-pro",
    ...
)

# Azure OpenAI (deployment name goes in llm_model)
agent = MongoAgent(
    name="assistant",
    llm_provider="azure_openai",
    llm_api_key="...",
    llm_model="my-deployment",
    azure_endpoint="https://your-resource.openai.azure.com/",
    ...
)

# Grove (LLM gateway)
agent = MongoAgent(name="assistant", llm_provider="grove", llm_api_key="...", ...)
```

## Session Management

```python
agent.end_session("user@example.com")

stats = agent.get_user_stats("user@example.com")
print(f"Conversations: {stats['conversation_count']}")
print(f"Short-term summaries: {stats['short_term_count']}")
print(f"Long-term facts: {stats['long_term_count']}")

agent.clear_user_memories("user@example.com", memory_type="conversation")
```

## Requirements

- Python 3.10+
- MongoDB Atlas cluster (M0, Flex, or M10+ with Compute Auto-Scale)
- API key for your chosen LLM provider (OpenAI, Azure OpenAI, Anthropic, Google, or Grove)

No Voyage AI API key required — Atlas manages embeddings for you.

## License

MIT
