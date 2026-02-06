"""
Neo4j Write Test Script
Verifies that the user can actually write data to the Neo4j database.
"""

from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv


def test_write():
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print(f"🔍 Testing Write Access to: {uri} (Database: {database})")

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password, database=database)

        # 1. Try a simple write query
        print("📝 Attempting to create a test node...")
        query = "CREATE (n:TestNode {name: 'PersistenceTest', timestamp: timestamp()}) RETURN id(n) as id"
        result = graph.query(query)

        if result:
            node_id = result[0]["id"]
            print(f"✅ Created test node with ID: {node_id}")

            # 2. Verify it exists
            print("👁️ Verifying node persistence...")
            verify_query = (
                f"MATCH (n:TestNode) WHERE id(n) = {node_id} RETURN n.name as name"
            )
            verify_result = graph.query(verify_query)

            if verify_result and verify_result[0]["name"] == "PersistenceTest":
                print("✅ Node verified! Write access is working.")

                # 3. Cleanup
                print("🧹 Cleaning up test node...")
                graph.query(f"MATCH (n:TestNode) WHERE id(n) = {node_id} DELETE n")
                print("✅ Cleanup complete.")
            else:
                print(
                    "❌ ERROR: Node was 'created' but could not be retrieved! This suggests transaction failed to commit or wrong database."
                )
        else:
            print("❌ ERROR: No result returned from CREATE query.")

    except Exception as e:
        print("\n" + "!" * 50)
        print("❌ WRITE FAILED")
        print("!" * 50)
        print(f"Error Message: {str(e)}")
        print("!" * 50)


if __name__ == "__main__":
    test_write()
