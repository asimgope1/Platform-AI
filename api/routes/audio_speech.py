from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from audio.service import AudioService

router = APIRouter(prefix="/v1/audio", tags=["Audio"])


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="TTS model alias")
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(default="alloy", description="Voice identifier (e.g. alloy, echo, fable, onyx, nova, shimmer)")
    response_format: str = Field(default="mp3", description="Audio format (mp3, wav, opus)")


@router.post("/speech")
async def create_speech(
    payload: SpeechRequest,
    context: TenantContext = Depends(get_current_tenant),
):
    """Text-to-speech audio generation endpoint."""
    context.check_scope("audio:speech")
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    service = AudioService()
    audio_bytes = await service.synthesize(
        text=payload.input,
        model=payload.model,
        voice=payload.voice,
        response_format=payload.response_format,
    )

    media_type = f"audio/{payload.response_format}" if payload.response_format != "mp3" else "audio/mpeg"
    return Response(content=audio_bytes, media_type=media_type)
