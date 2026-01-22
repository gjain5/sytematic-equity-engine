"""
FastAPI application entry point.

Design decisions:
- Minimal app setup with CORS enabled for local development
- Routes are separated into routes.py for cleaner organization
- No authentication (to be added later)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="Systematic Equity Engine API",
    description="API for systematic equity portfolio management",
    version="0.1.0",
)

# CORS middleware for local development and Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET"],  # Read-only API for now
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize any resources on startup."""
    pass  # Placeholder for future initialization


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    pass  # Placeholder for future cleanup
