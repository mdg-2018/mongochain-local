"""Main MongoAgent class for mongochain.

Embeddings are handled by MongoDB Atlas Automated Embedding (`autoEmbed`).
Callers do not manage a Voyage AI API key or generate vectors client-side.
"""

import json
from typing import Optional, Callable

from .config import AgentConfig, MemoryConfig
from .llm import LLMClient
from .memory import MemoryStore


class MongoAgent:
    """An AI agent with MongoDB-backed memory.

    Each agent gets its own MongoDB database with separate collections for
    different memory types. All memories are user-specific (by email/ID).

    Memory Types:
    - conversation_history: Raw chat log (90-day TTL)
    - short_term_memory: Session summaries (7-day TTL)
    - long_term_memory: Persistent user facts (no TTL)

    Vector search over each memory type uses Atlas Automated Embedding: Atlas
    generates and manages Voyage AI embeddings for the indexed text field.
    """

    def __init__(
        self,
        name: str,
        persona: str,
        mongo_uri: str,
        llm_api_key: str,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        collaborators: Optional[list[str]] = None,
        memory_config: Optional[MemoryConfig] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-15-preview",
        embedding_model: str = "voyage-4",
    ):
        """Initialize a MongoAgent.

        Args:
            name: Agent name (becomes the MongoDB database name)
            persona: The agent's personality/system prompt
            mongo_uri: MongoDB Atlas connection string
            llm_api_key: API key for the LLM provider
            llm_provider: "openai", "azure_openai", "anthropic", "google", or "grove"
            llm_model: Specific model to use (provider default if None).
                       For azure_openai, this is the deployment name.
            collaborators: List of agent names this agent can read memories from
            memory_config: Optional memory configuration
            azure_endpoint: Azure OpenAI endpoint URL (required for azure_openai)
            azure_api_version: Azure OpenAI API version
            embedding_model: Voyage AI model used by Atlas autoEmbed
                (default: "voyage-4"). Options: "voyage-4-lite", "voyage-4",
                "voyage-4-large", "voyage-code-3".
        """
        # Propagate embedding_model into memory_config if user didn't set one.
        if memory_config is None:
            memory_config = MemoryConfig(embedding_model=embedding_model)
        elif memory_config.embedding_model != embedding_model and embedding_model != "voyage-4":
            memory_config.embedding_model = embedding_model

        self._config = AgentConfig(
            name=name,
            persona=persona,
            mongo_uri=mongo_uri,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
            collaborators=collaborators or [],
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            embedding_model=embedding_model,
        )

        self.name = name
        self.persona = persona
        self.collaborators = list(self._config.collaborators)
        self._mongo_uri = mongo_uri

        self._llm = LLMClient(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )
        self._memory = MemoryStore(
            mongo_uri=mongo_uri,
            db_name=self._config.db_name,
            config=memory_config,
        )

        self._session_messages: dict[str, list] = {}
        self._tools: dict[str, dict] = {}

        self._setup()

    def _setup(self):
        status = self._memory.setup_collections()

        self._memory.save_agent_metadata(
            name=self.name,
            persona=self.persona,
            llm_provider=self._llm.provider,
            llm_model=self._llm.model,
            collaborators=self.collaborators,
            description=None,
        )

        print(f"Agent: {self.name} created. Check database '{self._config.db_name}' in your MongoDB cluster.")
        if status["collections_created"]:
            print(f"  Collections created: {', '.join(status['collections_created'])}")
        for vs in status["vector_index_status"]:
            print(f"  Vector index - {vs}")

    def chat(self, user_id: str, message: str) -> str:
        """Send a message and get a response."""
        if user_id not in self._session_messages:
            self._session_messages[user_id] = []

        self._memory.store_conversation(user_id=user_id, role="user", content=message)
        self._session_messages[user_id].append({"role": "user", "content": message})

        context = self._build_context(user_id, message)
        messages = self._build_messages(user_id, message, context)
        system_prompt = self._build_system_prompt(user_id, context)

        if self._tools:
            response = self._chat_with_tools(messages, system_prompt)
        else:
            response = self._llm.chat(messages, system_prompt=system_prompt)

        self._memory.store_conversation(user_id=user_id, role="assistant", content=response)
        self._session_messages[user_id].append({"role": "assistant", "content": response})

        self._extract_and_store_user_facts(user_id, message, response)

        if len(self._session_messages[user_id]) >= 10:
            self._update_short_term_summary(user_id)

        return response

    def _chat_with_tools(self, messages: list[dict], system_prompt: str) -> str:
        tools = [
            {"name": name, "description": tool["description"], "parameters": tool["parameters"]}
            for name, tool in self._tools.items()
        ]

        result = self._llm.chat_with_tools(messages, tools, system_prompt)

        if result["type"] == "tool_calls":
            tool_results = []
            for call in result["calls"]:
                tool_name = call["name"]
                tool_args = call["arguments"]
                if tool_name in self._tools:
                    try:
                        tool_result = self._tools[tool_name]["func"](**tool_args)
                        tool_results.append({"tool": tool_name, "success": True, "result": tool_result})
                    except Exception as e:
                        tool_results.append({"tool": tool_name, "success": False, "error": str(e)})
                else:
                    tool_results.append({"tool": tool_name, "success": False, "error": f"Tool '{tool_name}' is not available"})

            tool_summary = "\n".join(
                f"Tool '{tr['tool']}' result: {tr['result']}" if tr["success"]
                else f"Tool '{tr['tool']}' failed: {tr['error']}"
                for tr in tool_results
            )

            messages.append({"role": "assistant", "content": "I'll use multiple tools to help answer this."})
            messages.append({"role": "user", "content": f"Tool results:\n{tool_summary}"})
            return self._llm.chat(messages, system_prompt=system_prompt)

        if result["type"] == "tool_call":
            tool_name = result["name"]
            tool_args = result["arguments"]
            if tool_name in self._tools:
                try:
                    tool_result = self._tools[tool_name]["func"](**tool_args)
                    messages.append({"role": "assistant", "content": f"I'll use the {tool_name} tool to help answer this."})
                    messages.append({"role": "user", "content": f"Tool result from {tool_name}: {tool_result}"})
                    return self._llm.chat(messages, system_prompt=system_prompt)
                except Exception as e:
                    return f"I tried to use the {tool_name} tool but encountered an error: {str(e)}"
            return f"Tool '{tool_name}' was requested but is not available."

        return result["content"]

    def chat_stream(self, user_id: str, message: str):
        """Send a message and stream the response."""
        if user_id not in self._session_messages:
            self._session_messages[user_id] = []

        self._memory.store_conversation(user_id=user_id, role="user", content=message)
        self._session_messages[user_id].append({"role": "user", "content": message})

        context = self._build_context(user_id, message)
        messages = self._build_messages(user_id, message, context)

        full_response = []
        for chunk in self._llm.chat_stream(messages, system_prompt=self._build_system_prompt(user_id, context)):
            full_response.append(chunk)
            yield chunk

        response = "".join(full_response)
        self._memory.store_conversation(user_id=user_id, role="assistant", content=response)
        self._session_messages[user_id].append({"role": "assistant", "content": response})

        self._extract_and_store_user_facts(user_id, message, response)

        if len(self._session_messages[user_id]) >= 10:
            self._update_short_term_summary(user_id)

    def _build_context(self, user_id: str, query: str) -> dict:
        context = {
            "short_term": [],
            "long_term": [],
            "conversation": [],
            "collaborator_memories": {},
        }

        context["short_term"] = self._memory.search_short_term(user_id, query, limit=3)
        context["long_term"] = self._memory.search_long_term(user_id, query)

        conversation = self._memory.get_conversation_history(user_id, limit=10)
        context["conversation"] = [{"role": m["role"], "content": m["content"]} for m in conversation]

        for collab_name in self.collaborators:
            try:
                collab_memories = self._get_collaborator_memories(collab_name, user_id, query)
                if collab_memories:
                    context["collaborator_memories"][collab_name] = collab_memories
            except Exception:
                pass

        return context

    def _get_collaborator_memories(self, agent_name: str, user_id: str, query_text: str) -> list[dict]:
        collab_config = AgentConfig(
            name=agent_name,
            persona="",
            mongo_uri=self._mongo_uri,
            llm_api_key="",
        )
        collab_memory = MemoryStore(mongo_uri=self._mongo_uri, db_name=collab_config.db_name)
        try:
            return collab_memory.search_long_term(user_id, query_text, limit=3)
        except Exception:
            return collab_memory.get_user_long_term(user_id, limit=3)
        finally:
            collab_memory.close()

    def _build_system_prompt(self, user_id: str, context: dict) -> str:
        base_prompt = f"""You are an AI agent named {self.name}.

{self.persona}

You have access to memories about this user and past conversations.
When you learn something significant about the user (preferences, facts about them, their work, skills, etc.),
naturally incorporate this knowledge into your responses."""

        if context["long_term"]:
            facts = "\n".join(f"- {m['content']}" for m in context["long_term"])
            base_prompt += f"\n\nWhat you know about this user:\n{facts}"

        if context["short_term"]:
            summaries = "\n".join(f"- {m['summary']}" for m in context["short_term"])
            base_prompt += f"\n\nRecent interactions with this user:\n{summaries}"

        return base_prompt

    def _build_messages(self, user_id: str, current_message: str, context: dict) -> list[dict]:
        messages = []

        if context["collaborator_memories"]:
            collab_context = []
            for collab_name, memories in context["collaborator_memories"].items():
                if memories:
                    collab_context.append(
                        f"Information from {collab_name}:\n" +
                        "\n".join(f"- {m['content']}" for m in memories)
                    )
            if collab_context:
                messages.append({
                    "role": "user",
                    "content": "[Shared knowledge from other agents]\n" + "\n\n".join(collab_context),
                })
                messages.append({"role": "assistant", "content": "I'll incorporate this shared knowledge in my response."})

        for msg in (context["conversation"][:-1] if context["conversation"] else []):
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": current_message})
        return messages

    def _extract_and_store_user_facts(self, user_id: str, user_message: str, assistant_response: str):
        extraction_prompt = f"""Analyze this conversation exchange and extract any significant facts about the user.

User message: {user_message}
Assistant response: {assistant_response}

Extract facts about the user such as:
- Their job/role/profession
- Their company or organization
- Technical skills or expertise
- Preferences or interests
- Projects they're working on
- Personal details they shared

Return a JSON array of facts. Each fact should be a complete sentence.
If there are no significant facts, return an empty array: []

Example output: ["User works as a DevOps engineer", "User is interested in MongoDB optimization"]
Only return the JSON array, nothing else."""

        try:
            result = self._llm.chat(
                [{"role": "user", "content": extraction_prompt}],
                system_prompt="You are a fact extraction assistant. Only output valid JSON arrays.",
            )
            facts = json.loads(result.strip())
            if isinstance(facts, list) and facts:
                for fact in facts:
                    if isinstance(fact, str) and len(fact) > 10:
                        existing = self._memory.search_long_term(user_id, fact, limit=1)
                        # Only store if no near-duplicate exists (score < 0.9)
                        if not existing or existing[0].get("score", 0) < 0.9:
                            self._memory.store_long_term(
                                user_id=user_id,
                                content=fact,
                                category="user_fact",
                                metadata={"source": "auto_extracted"},
                            )
        except (json.JSONDecodeError, Exception):
            pass

    def _update_short_term_summary(self, user_id: str):
        if user_id not in self._session_messages or not self._session_messages[user_id]:
            return

        messages = self._session_messages[user_id]
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in messages[-10:]
        )

        summary_prompt = f"""Summarize this conversation session concisely.

{conversation_text}

Provide a JSON response with:
- "summary": A 1-2 sentence summary of what was discussed
- "topics": Array of main topics discussed
- "actions": Array of actions taken or outcomes

Example:
{{"summary": "Discussed MongoDB indexing strategies and helped debug a slow query.", "topics": ["MongoDB", "indexing", "performance"], "actions": ["explained compound indexes", "suggested query optimization"]}}

Only return valid JSON."""

        try:
            result = self._llm.chat(
                [{"role": "user", "content": summary_prompt}],
                system_prompt="You are a conversation summarizer. Only output valid JSON.",
            )
            summary_data = json.loads(result.strip())

            self._memory.store_short_term(
                user_id=user_id,
                summary=summary_data.get("summary", "Session summary"),
                topics_discussed=summary_data.get("topics", []),
                actions_taken=summary_data.get("actions", []),
            )
            self._session_messages[user_id] = []
        except (json.JSONDecodeError, Exception):
            pass

    # ==================== Public Methods ====================

    def set_persona(self, new_persona: str):
        self.persona = new_persona
        self._config.persona = new_persona
        self._memory.update_agent_persona(new_persona)
        print(f"Agent {self.name}'s persona updated.")

    def set_description(self, description: str):
        self._memory.save_agent_metadata(
            name=self.name,
            persona=self.persona,
            llm_provider=self._llm.provider,
            llm_model=self._llm.model,
            collaborators=self.collaborators,
            description=description,
        )
        print(f"Agent {self.name}'s description updated.")

    def add_collaborator(self, agent_name: str):
        if agent_name not in self.collaborators:
            self.collaborators.append(agent_name)
            self._memory.update_agent_collaborators(self.collaborators)
            print(f"Added {agent_name} as a collaborator for {self.name}.")

    def remove_collaborator(self, agent_name: str):
        if agent_name in self.collaborators:
            self.collaborators.remove(agent_name)
            self._memory.update_agent_collaborators(self.collaborators)
            print(f"Removed {agent_name} from collaborators for {self.name}.")

    def store_user_memory(self, user_id: str, content: str, category: str = "general"):
        """Manually store a fact about a user in long-term memory."""
        self._memory.store_long_term(
            user_id=user_id,
            content=content,
            category=category,
            metadata={"source": "manual"},
        )
        print(f"Memory stored for user {user_id} in category '{category}'.")

    def get_user_memories(self, user_id: str, category: Optional[str] = None, limit: int = 10) -> list[dict]:
        return self._memory.get_user_long_term(user_id, limit, category)

    def get_user_stats(self, user_id: str) -> dict:
        return self._memory.get_user_stats(user_id)

    def get_stats(self) -> dict:
        return self._memory.get_stats()

    def get_metadata(self) -> Optional[dict]:
        return self._memory.get_agent_metadata()

    # ==================== Tool Management ====================

    def register_tool(self, name: str, func: Callable, description: str, parameters: dict):
        """Register a custom tool/function for the agent to use."""
        self._tools[name] = {"func": func, "description": description, "parameters": parameters}
        self._memory.register_tool(name, description, parameters)
        print(f"Tool '{name}' registered for agent {self.name}.")

    def get_tools(self) -> list[dict]:
        return self._memory.get_tools()

    def remove_tool(self, name: str):
        if name in self._tools:
            del self._tools[name]
        if self._memory.remove_tool(name):
            print(f"Tool '{name}' removed from agent {self.name}.")
        else:
            print(f"Tool '{name}' not found.")

    def clear_user_memories(self, user_id: str, memory_type: Optional[str] = None):
        if memory_type == "conversation":
            self._memory.clear_user_conversation(user_id)
        elif memory_type == "short_term":
            self._memory.clear_user_short_term(user_id)
        elif memory_type == "long_term":
            self._memory.clear_user_long_term(user_id)
        else:
            self._memory.clear_user_all(user_id)
        print(f"Cleared {memory_type or 'all'} memories for user {user_id}.")

    def end_session(self, user_id: str):
        if user_id in self._session_messages and self._session_messages[user_id]:
            self._update_short_term_summary(user_id)
            print(f"Session ended for {user_id}. Summary saved to short-term memory.")

    def __repr__(self) -> str:
        return f"MongoAgent(name='{self.name}', provider='{self._llm.provider}')"
