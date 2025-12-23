#!/usr/bin/env python3
"""
EchoNotes Frontend Server
=========================

Simple HTTP server to serve the frontend.

Usage:
    python run_frontend.py              # Serve on port 3000
    python run_frontend.py --port 5000  # Custom port

Then open: http://localhost:3000
"""

import argparse
import http.server
import socketserver
import os
from pathlib import Path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS support"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve EchoNotes Frontend")
    parser.add_argument("--port", type=int, default=3000, help="Port to serve on")
    parser.add_argument("--host", default="localhost", help="Host to bind")
    args = parser.parse_args()
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗ ██████╗██╗  ██╗ ██████╗                     ║
    ║     ██╔════╝██╔════╝██║  ██║██╔═══██╗                    ║
    ║     █████╗  ██║     ███████║██║   ██║                    ║
    ║     ██╔══╝  ██║     ██╔══██║██║   ██║                    ║
    ║     ███████╗╚██████╗██║  ██║╚██████╔╝                    ║
    ║     ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                    ║
    ║                                                           ║
    ║     N O T E S   -   Frontend Server                      ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🌐 Frontend running at: http://{args.host}:{args.port}
    
    📋 Make sure the API server is running:
       python run_api.py --port 8000
    
    Press Ctrl+C to stop
    """)
    
    with socketserver.TCPServer((args.host, args.port), CORSHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Frontend server stopped")


if __name__ == "__main__":
    main()
