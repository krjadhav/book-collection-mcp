"""
HTTP-Based MCP Server for Book Collection
Exposes MCP via Server-Sent Events (SSE) over HTTP
"""
import os
import json
import asyncio
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.applications import Starlette
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Book Collection MCP Server")

# Add CORS middleware (allow all origins for easier access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP Server
mcp = Server("book-collection-server")
model = None
pc_index = None


def initialize():
    """Initialize embedding model and Pinecone connection"""
    global model, pc_index

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Pinecone...")
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    pc = Pinecone(api_key=api_key)
    pc_index = pc.Index("book-collection")

    print("✅ MCP Server initialized successfully!")


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_books",
            description="Search for books using natural language queries. Returns relevant books based on semantic similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'science fiction about space exploration', 'mystery novels by Agatha Christie')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_book_by_id",
            description="Get detailed information about a specific book by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The unique book ID"
                    }
                },
                "required": ["book_id"]
            }
        ),
        Tool(
            name="recommend_similar",
            description="Get book recommendations similar to a given book title",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the book to find similar books for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations (default: 5)",
                        "default": 5
                    }
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="search_by_author",
            description="Find all books by a specific author",
            inputSchema={
                "type": "object",
                "properties": {
                    "author": {
                        "type": "string",
                        "description": "Author name"
                    }
                },
                "required": ["author"]
            }
        ),
        Tool(
            name="get_collection_stats",
            description="Get statistics about the book collection including total count of books available",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""

    if name == "search_books":
        return await search_books(arguments)
    elif name == "get_book_by_id":
        return await get_book_by_id(arguments)
    elif name == "recommend_similar":
        return await recommend_similar(arguments)
    elif name == "search_by_author":
        return await search_by_author(arguments)
    elif name == "get_collection_stats":
        return await get_collection_stats(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_books(arguments: dict) -> list[TextContent]:
    """Search for books using semantic search"""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)

    query_embedding = model.encode(query).tolist()

    results = pc_index.query(
        vector=query_embedding,
        top_k=limit,
        include_metadata=True
    )

    if not results.matches:
        return [TextContent(
            type="text",
            text=f"No books found for query: '{query}'"
        )]

    books = []
    for match in results.matches:
        metadata = match.metadata
        books.append({
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "score": round(match.score, 3),
            "id": metadata.get("id", "")
        })

    response = f"Found {len(books)} books for query: '{query}'\n\n"
    for i, book in enumerate(books, 1):
        response += f"{i}. **{book['title']}** by {book['author']}\n"
        response += f"   Relevance: {book['score']}\n\n"

    return [TextContent(type="text", text=response)]


async def get_book_by_id(arguments: dict) -> list[TextContent]:
    """Get book details by ID"""
    book_id = arguments.get("book_id", "")

    result = pc_index.fetch(ids=[book_id])

    if not result.vectors or book_id not in result.vectors:
        return [TextContent(
            type="text",
            text=f"Book with ID '{book_id}' not found"
        )]

    metadata = result.vectors[book_id].metadata

    response = f"**{metadata.get('title', 'Unknown')}**\n\n"
    response += f"Author: {metadata.get('author', 'Unknown')}\n"
    response += f"ID: {book_id}\n"

    if metadata.get('cover_url'):
        response += f"\n![Cover]({metadata['cover_url']})\n"

    return [TextContent(type="text", text=response)]


async def recommend_similar(arguments: dict) -> list[TextContent]:
    """Recommend books similar to given title"""
    title = arguments.get("title", "")
    limit = arguments.get("limit", 5)

    query_embedding = model.encode(title).tolist()

    results = pc_index.query(
        vector=query_embedding,
        top_k=limit + 5,
        include_metadata=True
    )

    if not results.matches:
        return [TextContent(
            type="text",
            text=f"No books found similar to: '{title}'"
        )]

    recommendations = []
    for match in results.matches:
        book_title = match.metadata.get("title", "")
        if book_title.lower() != title.lower():
            recommendations.append({
                "title": book_title,
                "author": match.metadata.get("author", "Unknown"),
                "score": round(match.score, 3)
            })

        if len(recommendations) >= limit:
            break

    if not recommendations:
        return [TextContent(
            type="text",
            text=f"No recommendations found for: '{title}'"
        )]

    response = f"Books similar to '{title}':\n\n"
    for i, book in enumerate(recommendations, 1):
        response += f"{i}. **{book['title']}** by {book['author']}\n"
        response += f"   Similarity: {book['score']}\n\n"

    return [TextContent(type="text", text=response)]


async def search_by_author(arguments: dict) -> list[TextContent]:
    """Search books by author"""
    author = arguments.get("author", "")

    query = f"books by {author}"
    query_embedding = model.encode(query).tolist()

    results = pc_index.query(
        vector=query_embedding,
        top_k=50,
        include_metadata=True
    )

    books = []
    for match in results.matches:
        book_author = match.metadata.get("author", "")
        if author.lower() in book_author.lower():
            books.append({
                "title": match.metadata.get("title", "Unknown"),
                "author": book_author,
                "id": match.metadata.get("id", "")
            })

    if not books:
        return [TextContent(
            type="text",
            text=f"No books found by author: '{author}'"
        )]

    response = f"Found {len(books)} books by {author}:\n\n"
    for i, book in enumerate(books, 1):
        response += f"{i}. {book['title']}\n"

    return [TextContent(type="text", text=response)]


async def get_collection_stats(arguments: dict) -> list[TextContent]:
    """Get collection statistics from Pinecone"""
    stats = pc_index.describe_index_stats()

    total_vectors = stats.get('total_vector_count', 0)
    dimension = stats.get('dimension', 0)

    response = f"📚 **Book Collection Statistics**\n\n"
    response += f"**Total Books Available:** {total_vectors:,}\n"
    response += f"**Vector Dimension:** {dimension}\n"
    response += f"**Embedding Model:** all-MiniLM-L6-v2\n"
    response += f"**Search Method:** Semantic similarity (cosine)\n\n"
    response += f"You can search across all {total_vectors:,} books using natural language queries!"

    return [TextContent(type="text", text=response)]


# Create SSE transport
sse = SseServerTransport("/messages/")


# HTTP endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Book Collection MCP Server",
        "version": "1.0.0",
        "protocol": "mcp",
        "capabilities": ["tools"],
        "status": "ready"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


async def handle_sse(request: Request):
    """Handle MCP over SSE"""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp.run(
            streams[0],
            streams[1],
            mcp.create_initialization_options()
        )
    return Response()


async def handle_messages(scope, receive, send):
    """Handle incoming messages"""
    await sse.handle_post_message(scope, receive, send)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    initialize()


# Mount SSE and message endpoints
app.mount("/messages/", app=handle_messages)
app.add_api_route("/sse", handle_sse, methods=["GET"])


if __name__ == "__main__":
    import uvicorn

    # Run server
    port = int(os.getenv("PORT", 8001))
    print(f"🚀 Starting MCP HTTP Server on port {port}...")
    print(f"📚 Server will be available at: http://localhost:{port}")
    print(f"🔌 MCP endpoint: http://localhost:{port}/sse")

    uvicorn.run(app, host="0.0.0.0", port=port)
