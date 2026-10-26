from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from audio.service import AudioService

router = APIRouter(prefix="/v1/audio", tags=["Audio"])


@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    context: TenantContext = Depends(get_current_tenant),
):
    """Speech-to-text audio transcription endpoint."""
    context.check_scope("audio:transcribe")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file cannot be empty.")

    filename = file.filename or "audio.mp3"
    service = AudioService()
    result = await service.transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        model=model,
        language=language,
    )
    return result
