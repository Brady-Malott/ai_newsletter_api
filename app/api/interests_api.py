from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import UnitOfWork, get_current_user, get_uow
from app.api.schemas.interests_request_models import InterestExtractionRequest
from app.api.schemas.interests_response_models import InterestExtractionResponse
from app.core.errors import ErrorResponse, build_http_error
from app.core.prompt_sanitizer import PromptValidationError, sanitize_prompt
from app.core.rate_limit import (
    INTEREST_EXTRACT_RATE_LIMIT,
    limit,
    rate_limit_user_or_ip_key,
)
from app.db.models.user import User
from app.services.interest_service import InterestExtractionError
from app.strings import get_message_for_error

router = APIRouter()


@router.post(
    "/extract",
    response_model=InterestExtractionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
        status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
@limit(INTEREST_EXTRACT_RATE_LIMIT, key_func=rate_limit_user_or_ip_key)
async def extract_interests(
    request: Request,
    request_data: InterestExtractionRequest,
    _current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
) -> InterestExtractionResponse:
    """Extract interests from a natural language prompt."""
    try:
        sanitized_prompt = sanitize_prompt(request_data.prompt)
        result = await uow.interest_service.extract_interests_from_prompt(sanitized_prompt)
        return InterestExtractionResponse(
            add_interests=result.add_interests,
            remove_interests=result.remove_interests,
        )
    except PromptValidationError as exc:
        error_code, message = get_message_for_error(exc)
        raise build_http_error(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=error_code,
            message=message,
        ) from exc
    except InterestExtractionError as exc:
        if exc.error_code in ("llm_auth_failed", "llm_response_invalid"):
            status_code = status.HTTP_502_BAD_GATEWAY
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_code, message = get_message_for_error(exc)
        raise build_http_error(
            status_code=status_code,
            error=error_code,
            message=message,
        ) from exc
