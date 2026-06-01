"""Dependency that provides the authenticated user from the request."""

from __future__ import annotations

from fastapi import Depends, status

from app.api.dependencies.unit_of_work import UnitOfWork, get_uow
from app.core.auth import oauth2_scheme, verify_token
from app.core.errors import build_http_error
from app.db.models.user import User
from app.strings import get_error_code, get_message


async def get_current_user(
    token: str = Depends(oauth2_scheme), uow: UnitOfWork = Depends(get_uow)
) -> User:
    """FastAPI dependency to get the current authenticated user."""
    credentials_exception = build_http_error(
        status_code=status.HTTP_401_UNAUTHORIZED,
        error=get_error_code("auth.unauthorized"),
        message=get_message("auth.unauthorized"),
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    user_id_value = payload.get("sub")
    if not isinstance(user_id_value, str):
        raise credentials_exception

    try:
        user_id = int(user_id_value)
    except (ValueError, TypeError):
        raise credentials_exception from None

    user = await uow.auth_service.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception

    return user
