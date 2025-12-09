# api/controllers/transcript_controller.py
from fastapi import APIRouter
from ..schemas import Transcript
from ..services.dataset_service import save_transcript_to_dataset

router = APIRouter(prefix="/transcript", tags=["transcript"])


@router.post("/add")
async def add_transcript(data: Transcript):
    """Add a new transcript to the training dataset."""
    result = save_transcript_to_dataset(data.dict())
    return result

