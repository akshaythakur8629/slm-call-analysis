# api/server.py
from fastapi import FastAPI
from .controllers import transcript_controller, multitask_controller, gemma_controller
from .services.multitask_service import load_model_if_available
from .services.gemma_service import load_model

app = FastAPI(
    title="Call Analysis LLM API",
    description="API for training and analyzing call transcripts",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup if available."""
    load_model_if_available()
    load_model()


app.include_router(transcript_controller.router)
app.include_router(multitask_controller.router)
app.include_router(gemma_controller.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Call Analysis LLM API",
        "endpoints": {
            "transcript": "/transcript/add",
            "multitask": {
                "train": "/multitask/train",
                "analyze": "/multitask/analyze"
            },
            "gemma": {
                "train": "/gemma/train",
                "analyze": "/gemma/analyze"
            }
        }
    }
