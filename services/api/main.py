import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from platform_core.errors.exceptions import PlatformError
from platform_core.errors.api_errors import platform_error_handler, create_error_response
from api.middleware.request_id import RequestIdMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import health, models, chat_completions, embeddings, documents, audio_transcriptions, audio_speech
from services.api.lifespan import lifespan


def create_app() -> FastAPI:
    app = FastAPI(
        title="Standalone AI Platform API",
        description="Provider-agnostic AI platform with OpenAI-compatible endpoints, RAG, and safe tool calling.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware setup (LIFO order in Starlette)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, max_requests_per_minute=300)
    app.add_middleware(RequestIdMiddleware)

    # Global Exception Handlers
    @app.exception_handler(PlatformError)
    async def custom_platform_error_handler(request: Request, exc: PlatformError):
        return await platform_error_handler(request, exc)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return create_error_response(
            message=f"Internal server error: {str(exc)}",
            code="internal_server_error",
            status_code=500,
        )

    # Mount Routes
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat_completions.router)
    app.include_router(embeddings.router)
    app.include_router(documents.router)
    app.include_router(audio_transcriptions.router)
    app.include_router(audio_speech.router)

    # Mount Interactive Test UI Console
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        @app.get("/", response_class=FileResponse, include_in_schema=False)
        @app.get("/ui", response_class=FileResponse, include_in_schema=False)
        async def serve_ui():
            return FileResponse(static_file)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=True)
