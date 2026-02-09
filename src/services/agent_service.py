"""
AgentService - Provides an agentic interface to the knowledge graph.
Uses LangChain to reason about questions and select appropriate tools.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from src.services.interfaces import INeo4jRepository, ITraversalService
from src import config


class AgentService:
    """
    Service that runs a LangChain agent to answer questions using
    graph tools (Vector Search, Traversal, etc.).
    """

    def __init__(
        self, neo4j_repo: INeo4jRepository, traversal_service: ITraversalService
    ):
        self.neo4j_repo = neo4j_repo
        self.traversal_service = traversal_service
        self._agent_graph = None

    def _initialize_agent(self):
        """Lazy initialization of the agent executor."""
        if self._agent_graph:
            return

        llm = ChatOpenAI(
            model=config.LLM_MODEL, temperature=0, api_key=config.OPENAI_API_KEY
        )

        # Define Tools
        @tool
        def search_vector_store(query: str, top_k: int = 5) -> str:
            """
            Search for entities in the knowledge graph using semantic similarity.
            Use this to find starting points or when unique names are not known.
            Returns a list of found entities with their descriptions and scores.
            """
            results = self.neo4j_repo.vector_search(query, top_k=top_k)
            if not results:
                return "No matching entities found."

            output = []
            for r in results:
                output.append(
                    f"ID: {r['id']}, Name: {r.get('name', 'Unknown')}, Description: {r.get('description', 'N/A')}, Score: {r['score']:.2f}"
                )
            return "\n".join(output)

        @tool
        def get_entity_context(entity_id: str) -> str:
            """
            Get detailed context about a specific entity, including its neighbors and relationships.
            Use this when you have a specific entity ID (often from search_vector_store) and need to know more about it.
            """
            context = self.traversal_service.get_entity_context(entity_id)
            if not context.get("entity"):
                return f"Entity with ID '{entity_id}' not found."

            return context.get("context_summary", "No context available.")

        @tool
        def find_connections(entity_a: str, entity_b: str) -> str:
            """
            Find the shortest path connection between two entities.
            Useful for "how is X related to Y" questions.
            Inputs should be entity IDs or precise names.
            """
            # Try to resolve names to IDs first if they don't look like IDs
            id_a = self._resolve_to_id(entity_a)
            id_b = self._resolve_to_id(entity_b)

            if not id_a:
                return f"Could not find entity '{entity_a}'."
            if not id_b:
                return f"Could not find entity '{entity_b}'."

            result = self.traversal_service.find_connections(id_a, id_b)
            if result["connected"]:
                return f"Connected: {result['description']}"
            else:
                return f"No connection found between {entity_a} and {entity_b}."

        @tool
        def search_documents(query: str) -> str:
            """
            Search the original source documents (chunks).
            Use this to get specific text details that might not be in the graph structure.
            """
            results = self.neo4j_repo.vector_search_documents(query, top_k=3)
            if not results:
                return "No relevant document chunks found."

            output = []
            for r in results:
                content = r.get("content", "")
                # Truncate if too long to save context window
                if len(content) > 500:
                    content = content[:500] + "..."
                output.append(f"Content: {content}")
            return "\n\n".join(output)

        tools = [
            search_vector_store,
            get_entity_context,
            find_connections,
            search_documents,
        ]

        # Create Agent (LangGraph based)
        self._agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""You are a helpful AI assistant with access to a Knowledge Graph about Generative AI.
            
            Your goal is to answer the user's question by gathering information from the graph.
            
            GUIDELINES:
            1. ERROR HANDLING: If you search for an entity and don't find it, try a different variation of the name or use `search_vector_store` again with a broader query.
            2. ID USAGE: The tools often require specific Entity IDs. Use `search_vector_store` to find the correct ID and Name for an entity first.
            3. MULTI-STEP REASONING: 
               - If asked about "Model A", first find "Model A".
               - Then look at its context or specific relationships.
            4. CITATIONS: When you find information, mention the source nodes or relationships.
            
            Do not guess. If you cannot find the answer in the graph or documents, say so.
            """,
        )
        self._agent_executor = (
            self._agent_graph
        )  # Validation alias if needed, but we use _agent_graph separatey

    def _resolve_to_id(self, name_or_id: str) -> Optional[str]:
        """Helper to resolve a name to an ID using vector search if needed."""
        # Simple heuristic: if it has spaces, it's likely a name, not an ID
        # But we'll just check if it exists directly first (TODO: Need a check_exists method)

        # For now, just try vector search top 1
        results = self.neo4j_repo.vector_search(name_or_id, top_k=1)
        if results:
            return results[0]["id"]
        return None

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Run the agent to answer a question.
        Returns a dictionary with the answer and the thought process (intermediate steps).
        """
        self._initialize_agent()

        try:
            # invoke returns a dict with 'messages'
            inputs = {"messages": [HumanMessage(content=question)]}
            result = self._agent_graph.invoke(inputs)

            # The last message should be the AIMessage with the answer
            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None

            answer = last_message.content if last_message else "No answer generated."

            # Extract sources from ToolMessages
            sources = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    if msg.name == "search_documents":
                        # Naive parsing of document search results
                        content = msg.content
                        if "Content:" in content:
                            parts = content.split("Content:")
                            for part in parts[1:]:
                                clean_part = part.strip()[:200] + "..."
                                sources.append(
                                    {"type": "document", "content": clean_part}
                                )
                    elif msg.name == "search_vector_store":
                        # Naive parsing of entity search results
                        lines = msg.content.split("\n")
                        for line in lines:
                            if "Name:" in line:
                                sources.append(
                                    {"type": "entity", "content": line.strip()}
                                )
                    elif msg.name == "get_entity_context":
                        sources.append(
                            {
                                "type": "context",
                                "content": f"Context for {msg.tool_call_id}",
                            }
                        )  # Placeholder parsing

            return {"answer": answer, "sources": sources, "status": "success"}
        except Exception as e:
            return {
                "answer": f"I encountered an error while trying to answer that: {str(e)}",
                "status": "error",
                "error": str(e),
            }
