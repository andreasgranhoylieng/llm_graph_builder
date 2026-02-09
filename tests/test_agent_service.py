import pytest
from unittest.mock import MagicMock, patch
from src.services.agent_service import AgentService
from src.services.interfaces import INeo4jRepository, ITraversalService
from langchain_core.messages import AIMessage, ToolMessage


@pytest.fixture
def mock_neo4j_repo():
    return MagicMock(spec=INeo4jRepository)


@pytest.fixture
def mock_traversal_service():
    return MagicMock(spec=ITraversalService)


@pytest.fixture
def agent_service(mock_neo4j_repo, mock_traversal_service):
    return AgentService(mock_neo4j_repo, mock_traversal_service)


def test_resolve_to_id_with_search_results(agent_service, mock_neo4j_repo):
    # Setup
    mock_neo4j_repo.vector_search.return_value = [{"id": "model-123", "name": "GPT-4"}]

    # Execute
    result = agent_service._resolve_to_id("GPT-4")

    # Verify
    assert result == "model-123"
    mock_neo4j_repo.vector_search.assert_called_once_with("GPT-4", top_k=1)


def test_resolve_to_id_no_results(agent_service, mock_neo4j_repo):
    # Setup
    mock_neo4j_repo.vector_search.return_value = []

    # Execute
    result = agent_service._resolve_to_id("NonExistent")

    # Verify
    assert result is None


@patch("src.services.agent_service.ChatOpenAI")
@patch("src.services.agent_service.create_agent")
def test_ask_success(mock_create_agent, mock_chat, agent_service):
    # Setup
    mock_graph = mock_create_agent.return_value

    # Mock tool messages for source extraction
    tool_msg_docs = ToolMessage(
        content="Content: Doc 1 info", tool_call_id="1", name="search_documents"
    )
    tool_msg_ents = ToolMessage(
        content="ID: 1, Name: Entity A", tool_call_id="2", name="search_vector_store"
    )
    ai_msg = AIMessage(content="The answer is based on Doc 1 and Entity A.")

    # Return full conversation history
    mock_graph.invoke.return_value = {
        "messages": [tool_msg_docs, tool_msg_ents, ai_msg]
    }

    # Execute
    response = agent_service.ask("What is the answer?")

    # Verify
    assert response["status"] == "success"
    assert response["answer"] == "The answer is based on Doc 1 and Entity A."

    # Verify sources
    sources = response.get("sources", [])
    assert len(sources) >= 2
    assert any(s["type"] == "document" and "Doc 1" in s["content"] for s in sources)
    assert any(s["type"] == "entity" and "Entity A" in s["content"] for s in sources)

    mock_graph.invoke.assert_called_once()


@patch("src.services.agent_service.ChatOpenAI")
@patch("src.services.agent_service.create_agent")
def test_ask_error(mock_create_agent, mock_chat, agent_service):
    # Setup
    mock_graph = mock_create_agent.return_value
    mock_graph.invoke.side_effect = Exception("API Error")

    # Execute
    response = agent_service.ask("What is the answer?")

    # Verify
    assert response["status"] == "error"
    assert "API Error" in response["error"]
