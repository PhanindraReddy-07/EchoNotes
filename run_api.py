#!/usr/bin/env python3
"""
EchoNotes API Server
====================

Run the FastAPI server for EchoNotes.

Usage:
    python run_api.py                    # Default: localhost:8000
    python run_api.py --port 5000        # Custom port
    python run_api.py --host 0.0.0.0     # Allow external access
    python run_api.py --reload           # Development mode with auto-reload

API Documentation:
    http://localhost:8000/docs   - Swagger UI
    http://localhost:8000/redoc  - ReDoc
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run EchoNotes API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_api.py                      # Run on localhost:8000
    python run_api.py --port 5000          # Run on port 5000
    python run_api.py --host 0.0.0.0       # Allow external connections
    python run_api.py --reload             # Auto-reload on code changes
    python run_api.py --workers 4          # Use 4 workers (production)
        """
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1, use 0.0.0.0 for external)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗ ██████╗██╗  ██╗ ██████╗                     ║
    ║     ██╔════╝██╔════╝██║  ██║██╔═══██╗                    ║
    ║     █████╗  ██║     ███████║██║   ██║                    ║
    ║     ██╔══╝  ██║     ██╔══██║██║   ██║                    ║
    ║     ███████╗╚██████╗██║  ██║╚██████╔╝                    ║
    ║     ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                    ║
    ║                                                           ║
    ║     N O T E S   -   API Server                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Starting EchoNotes API Server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Reload: {'enabled' if args.reload else 'disabled'}")
    print()
    print(f"📚 API Documentation:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   ReDoc:      http://{args.host}:{args.port}/redoc")
    print()
    print(f"🔗 Endpoints:")
    print(f"   POST /api/upload          - Upload audio file")
    print(f"   POST /api/transcribe      - Transcribe audio")
    print(f"   POST /api/analyze         - Analyze text")
    print(f"   POST /api/generate        - Generate document")
    print(f"   POST /api/process         - Full pipeline")
    print(f"   GET  /api/status/{{job_id}} - Check job status")
    print(f"   GET  /api/download/{{job_id}} - Download document")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        import uvicorn
        
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level
        )
    except ImportError:
        print("❌ Error: uvicorn not installed")
        print("   Install with: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()
