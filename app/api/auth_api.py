from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import UnitOfWork, get_current_user, get_uow
from app.api.schemas.auth_request_models import LoginUserRequest, RegisterUserRequest
from app.api.schemas.auth_response_models import (
    AccessTokenResponse,
    DeleteUserResponse,
    UserResponse,
)
from app.core.auth import create_access_token
from app.core.errors import ErrorResponse, build_http_error
from app.core.rate_limit import (
    AUTH_DELETE_RATE_LIMIT,
    AUTH_LOGIN_RATE_LIMIT,
    AUTH_REGISTER_RATE_LIMIT,
    limit,
    rate_limit_ip_key,
    rate_limit_user_or_ip_key,
)
from app.db.models.user import User
from app.services.auth_service import (
    AuthenticationError,
    InvalidCredentialsError,
    PasswordTooLongError,
    UserAlreadyExistsError,
)
from app.strings import get_message_for_error

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
    },
)
@limit(AUTH_REGISTER_RATE_LIMIT, key_func=rate_limit_ip_key)
async def register(
    request: Request,
    user_data: RegisterUserRequest,
    uow: UnitOfWork = Depends(get_uow),
) -> UserResponse:
    """Register a new user."""
    try:
        new_user = await uow.auth_service.register_user(user_data.email, user_data.password)
        return UserResponse.model_validate(new_user)
    except AuthenticationError as e:
        if isinstance(e, UserAlreadyExistsError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(e, PasswordTooLongError):
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        error_code, message = get_message_for_error(e)
        raise build_http_error(
            status_code=status_code,
            error=error_code,
            message=message,
        ) from e


@router.post(
    "/login",
    response_model=AccessTokenResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
    },
)
@limit(AUTH_LOGIN_RATE_LIMIT, key_func=rate_limit_ip_key)
async def login(
    request: Request,
    credentials: LoginUserRequest,
    uow: UnitOfWork = Depends(get_uow),
) -> AccessTokenResponse:
    """Authenticate user and return JWT token."""
    try:
        user = await uow.auth_service.authenticate_user(credentials.email, credentials.password)
    except AuthenticationError as e:
        if isinstance(e, InvalidCredentialsError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(e, PasswordTooLongError):
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        headers = (
            {"WWW-Authenticate": "Bearer"} if status_code == status.HTTP_401_UNAUTHORIZED else None
        )
        error_code, message = get_message_for_error(e)
        raise build_http_error(
            status_code=status_code,
            error=error_code,
            message=message,
            headers=headers,
        ) from e

    access_token = create_access_token(data={"sub": str(user.id)})

    return AccessTokenResponse(access_token=access_token)


@router.get(
    "/me",
    response_model=UserResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
    },
)
async def get_me(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse.model_validate(current_user)


@router.delete(
    "/me",
    response_model=DeleteUserResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
    },
)
@limit(AUTH_DELETE_RATE_LIMIT, key_func=rate_limit_user_or_ip_key)
async def delete_me(
    request: Request,
    current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
) -> DeleteUserResponse:
    """Delete the current authenticated user and all associated data."""
    deleted_id = await uow.auth_service.delete_user(current_user.id)
    return DeleteUserResponse(deleted_user_id=deleted_id)
