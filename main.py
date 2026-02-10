"""
Neo4j LLM Knowledge Graph Builder - CLI Entry Point

This tool builds a knowledge graph from PDF/TXT documents using LLM-based
entity and relationship extraction, storing results in Neo4j.

Features:
- Large-scale batch processing with checkpointing
- Progress tracking with ETA
- Rate limiting for API calls
- Automatic retry on failures
"""

import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.controllers.graph_controller import GraphController


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("     🧠 NEO4J LLM KNOWLEDGE GRAPH BUILDER")
    print("     Large-Scale Document Processing Edition")
    print("=" * 60)


def print_menu():
    """Print main menu options."""
    print("\n--- Main Menu ---")
    print("1. 🚀 Build Graph (Batch Mode - Recommended)")
    print("2. 📋 Preview Folder (See what will be processed)")
    print("3. 📊 View Processing Status")
    print("4. 🔄 Retry Failed Files")
    print("5. 🗑️  Reset Job (Start Fresh)")
    print("6. 💾 View Database Statistics")
    print("7. 💬 Chat with your Graph")
    print("8. ⚙️  Build Graph (Legacy Mode - Small Datasets)")
    print("9. 🕵️ Chat with detailed logging")
    print("10. ❌ Exit")


def get_folder_path() -> str:
    """Get folder path from user."""
    folder = input("\nEnter folder path (or press Enter for ./data): ").strip()
    return folder if folder else "./data"


def main():
    """Main entry point."""
    controller = GraphController()

    print_header()

    while True:
        print_menu()
        choice = input("\nSelect an option (1-10): ").strip()

        if choice == "1":
            # Batch processing mode
            folder = get_folder_path()

            if not os.path.exists(folder):
                print(f"❌ Folder not found: {folder}")
                continue

            resume = (
                input("Resume from checkpoint if available? (Y/n): ").strip().lower()
                != "n"
            )

            try:
                result = controller.process_folder_batch(folder, resume=resume)
                print(result)
            except KeyboardInterrupt:
                print("\n⚠️ Processing interrupted. Progress saved.")
            except Exception as e:
                print(f"\n❌ Error: {e}")

        elif choice == "2":
            # Preview folder
            folder = get_folder_path()

            if not os.path.exists(folder):
                print(f"❌ Folder not found: {folder}")
                continue

            print(controller.preview_folder(folder))

        elif choice == "3":
            # View processing status
            status = controller.get_processing_status()

            if status.get("status") == "no_job":
                print("\n📋 No processing job found.")
            else:
                print("\n" + "=" * 50)
                print("📊 PROCESSING STATUS")
                print("=" * 50)
                print(f"Status: {status['status']}")
                print(
                    f"Progress: {status['processed_files']}/{status['total_files']} files ({status['progress_percentage']:.1f}%)"
                )
                print(f"Failed: {status['failed_files']}")
                print(f"Chunks processed: {status['total_chunks']:,}")
                print(f"Nodes created: {status['total_nodes']:,}")
                print(f"Relationships: {status['total_relationships']:,}")
                print(f"Started: {status['start_time']}")
                print(f"Last checkpoint: {status['last_checkpoint']}")
                if status["current_file"]:
                    print(
                        f"Currently processing: {os.path.basename(status['current_file'])}"
                    )
                print("=" * 50)

        elif choice == "4":
            # Retry failed files
            confirm = input("Retry all failed files? (y/N): ").strip().lower()
            if confirm == "y":
                result = controller.retry_failed_files()
                print(result)

        elif choice == "5":
            # Reset job
            confirm = (
                input("⚠️ This will delete all progress. Are you sure? (y/N): ")
                .strip()
                .lower()
            )
            if confirm == "y":
                result = controller.reset_job()
                print(result)

        elif choice == "6":
            # Database statistics
            stats = controller.get_database_stats()

            print("\n" + "=" * 50)
            print("💾 DATABASE STATISTICS")
            print("=" * 50)

            if "error" in stats:
                print(f"❌ Error: {stats['error']}")
            else:
                print(f"Total Nodes: {stats.get('total_nodes', 0):,}")
                print(f"Total Relationships: {stats.get('total_relationships', 0):,}")

                if stats.get("label_counts"):
                    print("\nNodes by Label:")
                    for item in stats["label_counts"][:10]:
                        labels = item.get("labels", ["Unknown"])
                        count = item.get("count", 0)
                        print(f"  - {', '.join(labels)}: {count:,}")

            print("=" * 50)

        elif choice == "7":
            # Chat mode with hybrid search
            print("\n💬 Chat Mode - Ask questions about your knowledge graph")
            print(
                "Commands: 'back' = menu, 'explore <entity>' = show details, 'connect <a> to <b>' = find path"
            )
            print("Using hybrid vector + graph search for better results\n")

            while True:
                q = input("You: ").strip()
                if q.lower() == "back":
                    break
                if not q:
                    continue

                # Handle special commands
                if q.lower().startswith("explore "):
                    entity_name = q[8:].strip()
                    result = controller.explore_entity(entity_name)
                    if result.get("entity"):
                        print(
                            f"\n📊 Entity: {result['entity'].get('name', entity_name)}"
                        )
                        print(f"   Labels: {result['entity'].get('labels', [])}")
                        print(
                            f"   Description: {result['entity'].get('description', 'N/A')}"
                        )
                        if result.get("neighbors"):
                            print(
                                f"\n🔗 Connected to {len(result['neighbors'])} entities:"
                            )
                            for n in result["neighbors"][:5]:
                                print(f"   - {n.get('name', n.get('id', 'unknown'))}")
                    else:
                        print(f"\n❌ {result.get('message', 'Entity not found')}")
                    print()
                    continue

                if q.lower().startswith("connect "):
                    # Parse "connect A to B"
                    parts = q[8:].split(" to ")
                    if len(parts) == 2:
                        result = controller.find_connections(
                            parts[0].strip(), parts[1].strip()
                        )
                        if result.get("connected"):
                            print(f"\n✅ Connection found ({result['hops']} hops):")
                            print(f"   {result['description']}")
                        else:
                            print(
                                f"\n❌ {result.get('description', 'No connection found')}"
                            )
                    else:
                        print("Usage: connect <entity A> to <entity B>")
                    print()
                    continue

                # Agentic Chat (Clean Mode)
                result = controller.chat_agent(q)

                if "error" in result:
                    print(f"\n❌ {result['error']}\n")
                else:
                    print(f"\n🤖 Answer: {result['answer']}")

                    # Show sources if available
                    if result.get("sources"):
                        print("\n📚 Sources:")
                        for source in result["sources"][:3]:
                            content = source.get("content", "").strip()
                            if len(content) > 100:
                                content = content[:100] + "..."
                            print(f"   - [{source.get('type', 'unknown')}] {content}")
                    print()

        elif choice == "8":
            # Legacy mode
            folder = get_folder_path()

            print("\n⚠️ Legacy mode loads all files into memory.")
            print("Use Batch Mode (option 1) for large datasets.\n")

            confirm = input("Continue with legacy mode? (y/N): ").strip().lower()
            if confirm == "y":
                result = controller.process_folder(folder)
                print(f"\n{result}")

        elif choice == "9":
            # Detailed Logging Chat
            print("\n" + "=" * 60)
            print("🕵️ CHAT WITH DETAILED LOGGING")
            print("=" * 60)
            print("Type 'exit' or 'quit' to return to menu.")

            while True:
                question = input("\n📝 Ask a question: ").strip()

                if question.lower() in ["exit", "quit"]:
                    break

                if not question:
                    continue

                print("⏳ Thinking and gathering logs...")
                response = controller.chat_with_logging(question)

                print(f"\n💡 Answer: {response.get('answer')}")

                trace = response.get("execution_trace")
                if trace:
                    print(f"\n📜 Execution Trace ({len(trace)} steps):")
                    for step in trace:
                        print(f"  [{step['step']}] {step['type'].upper()}")
                        if "tool_calls" in step:
                            for tc in step["tool_calls"]:
                                print(f"      🛠️ Tool: {tc['name']} -> {tc['args']}")
                        if "content" in step:
                            # Safely truncating content for display
                            content = str(step["content"])
                            if len(content) > 200:
                                content = content[:200] + "..."
                            print(f"      📝 {content}")

                if "error" in response:
                    print(f"❌ Error: {response['error']}")

        elif choice == "10":
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid choice. Please select 1-10.")


if __name__ == "__main__":
    main()
