import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.controllers.graph_controller import GraphController


def test_verbose_chat():
    print("Initializing GraphController...")
    controller = GraphController()

    question = "connect pokemon red to elon musk"
    print(f"\nAsking question: '{question}' with logging...")

    # Test get_connections (Option 7 logic)
    print("\nTesting find_connections (Option 7 logic)...")
    result = controller.find_connections("Pokemon Red", "Elon Musk")
    print(f"Result: {result}")


if __name__ == "__main__":
    test_verbose_chat()
