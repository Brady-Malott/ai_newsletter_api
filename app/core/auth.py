from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token extractor (OAuth2PasswordBearer is FastAPI's helper for extracting
# Bearer tokens from Authorization headers - we're using JWT, not OAuth2)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    # bcrypt only considers the first 72 bytes of the password; enforce this to avoid
    # silent truncation and align with unit tests that expect a ValueError.
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password must not exceed 72 bytes when UTF-8 encoded.")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    # Mirror the same 72-byte limit check used when hashing to avoid verifying
    # truncated passwords as valid.
    if len(plain_password.encode("utf-8")) > 72:
        raise ValueError("Password must not exceed 72 bytes when UTF-8 encoded.")
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        return None
