# api/controllers/multitask_controller.py
import subprocess
import sys
from fastapi import APIRouter, HTTPException
from ..services.multitask_service import (
    load_model_if_available,
    analyze_transcript,
    is_model_loaded,
)

router = APIRouter(prefix="/multitask", tags=["multitask"])


@router.post("/train")
async def train_model():
    """
    Trigger local training for multitask model (blocking).
    For small data this is fine.
    """
    result = subprocess.run(
        [sys.executable, "training/train_model.py"],
        capture_output=True,
        text=True,
    )
    
    # Reload model after training
    load_model_if_available()
    
    return {
        "status": "training_finished",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@router.post("/analyze")
async def analyze(payload: dict):
    """
    Analyze transcript using multitask model.
    Input:
    {
      "transcript": "AGENT: ...\nUSER: ..."
    }
    """
    transcript_text = payload.get("transcript", "")
    
    if not transcript_text:
        raise HTTPException(status_code=400, detail="transcript field is required")
    
    try:
        result = analyze_transcript(transcript_text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

