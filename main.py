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
    print("9. ❌ Exit")


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
        choice = input("\nSelect an option (1-9): ").strip()

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
            # Chat mode
            print("\n💬 Chat Mode - Ask questions about your knowledge graph")
            print("Type 'back' to return to menu\n")

            while True:
                q = input("You: ").strip()
                if q.lower() == "back":
                    break
                if not q:
                    continue

                answer = controller.chat(q)
                print(f"\n🤖 Answer: {answer}\n")

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
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print("❌ Invalid choice. Please select 1-9.")


if __name__ == "__main__":
    main()
