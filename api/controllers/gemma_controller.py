# api/controllers/gemma_controller.py
import subprocess
import sys
from fastapi import APIRouter, HTTPException
from ..services.gemma_service import (
    load_model,
    analyze_transcript,
    is_model_loaded,
)

router = APIRouter(prefix="/gemma", tags=["gemma"])


@router.post("/train")
async def train_gemma():
    """
    Trigger local training for Gemma model (blocking).
    """
    result = subprocess.run(
        [sys.executable, "model/finetune_gemma.py"],
        capture_output=True,
        text=True
    )
    
    load_model()
    
    return {
        "status": "Gemma training finished",
        "stdout": result.stdout,
        "stderr": result.stderr
    }


@router.post("/analyze")
async def analyze_llm(payload: dict):
    """
    Analyze transcript using Gemma model.
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

