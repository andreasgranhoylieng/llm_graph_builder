"""
Neo4j Connectivity Test Script - Comprehensive Diagnostics
Tries multiple connection methods to find a working approach.
"""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv


def test_connection():
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    print("=" * 60)
    print("NEO4J CONNECTION DIAGNOSTICS")
    print("=" * 60)
    print(f"URI: {uri}")
    print(f"User: {user}")
    print("=" * 60)

    if not all([uri, user, password]):
        print("ERROR: Missing Neo4j credentials in .env file")
        return

    results = {}

    # Method 1: Standard neo4j+s:// (strict SSL)
    print("\n--- Method 1: Standard Connection (neo4j+s://) ---")
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
        print("SUCCESS!")
        results["neo4j+s://"] = True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        results["neo4j+s://"] = False

    # Method 2: Self-signed cert (neo4j+ssc://) - bypasses SSL verification
    print("\n--- Method 2: Self-Signed Cert (neo4j+ssc://) ---")
    ssc_uri = uri.replace("neo4j+s://", "neo4j+ssc://")
    print(f"Trying: {ssc_uri}")
    try:
        with GraphDatabase.driver(ssc_uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
        print("SUCCESS!")
        results["neo4j+ssc://"] = True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        results["neo4j+ssc://"] = False

    # Method 3: Bolt protocol (bolt+s://)
    print("\n--- Method 3: Bolt Protocol (bolt+s://) ---")
    bolt_uri = uri.replace("neo4j+s://", "bolt+s://")
    print(f"Trying: {bolt_uri}")
    try:
        with GraphDatabase.driver(bolt_uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
        print("SUCCESS!")
        results["bolt+s://"] = True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        results["bolt+s://"] = False

    # Method 4: Bolt with self-signed cert (bolt+ssc://)
    print("\n--- Method 4: Bolt + Self-Signed (bolt+ssc://) ---")
    bolt_ssc_uri = uri.replace("neo4j+s://", "bolt+ssc://")
    print(f"Trying: {bolt_ssc_uri}")
    try:
        with GraphDatabase.driver(bolt_ssc_uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
        print("SUCCESS!")
        results["bolt+ssc://"] = True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        results["bolt+ssc://"] = False

    # Method 5: With explicit port
    print("\n--- Method 5: Explicit Port 7687 ---")
    if ":7687" not in uri:
        port_uri = uri.replace(".io", ".io:7687")
        print(f"Trying: {port_uri}")
        try:
            with GraphDatabase.driver(port_uri, auth=(user, password)) as driver:
                driver.verify_connectivity()
            print("SUCCESS!")
            results["explicit_port"] = True
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            results["explicit_port"] = False
    else:
        print("Skipped (port already in URI)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working_methods = [k for k, v in results.items() if v]

    if working_methods:
        print(f"WORKING METHODS: {', '.join(working_methods)}")
        print("\nRecommendation: Update your .env to use the first working protocol.")
    else:
        print("NO METHODS WORKED!")
        print("\nPossible causes:")
        print("  1. Neo4j Aura instance is PAUSED or OFFLINE")
        print("     -> Check https://console.neo4j.io/ and RESUME the instance")
        print("  2. Instance was DELETED or EXPIRED")
        print("     -> Create a new instance in Neo4j Aura")
        print("  3. Password is incorrect")
        print("     -> Reset password in Neo4j Aura console")
        print("  4. Database ID is wrong")
        print("     -> Verify the ID in the connection string")

    print("=" * 60)


if __name__ == "__main__":
    test_connection()
