"""Request models for auth API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.strings import get_message


def _validate_password_length(password: str) -> str:
    """Validate that password does not exceed 72 bytes when UTF-8 encoded.

    We use a 50 character limit to stay under the 72-byte bcrypt limit
    with multi-byte UTF-8 characters.
    """
    if len(password.encode("utf-8")) > 72:
        raise ValueError(get_message("auth.password_too_long", max_length=50))
    return password


class RegisterUserRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(..., max_length=320, description="Valid email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=50,
        description="Password between 8 and 50 characters",
    )
    confirm_password: str = Field(
        ...,
        min_length=8,
        max_length=50,
        description="Password confirmation (must match password)",
    )

    @field_validator("password", "confirm_password")
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        return _validate_password_length(v)

    @model_validator(mode="after")
    def passwords_match(self) -> RegisterUserRequest:
        if self.password != self.confirm_password:
            raise ValueError(get_message("auth.passwords_mismatch"))
        return self


class LoginUserRequest(BaseModel):
    """Request model for user login."""

    email: EmailStr = Field(..., max_length=320, description="Valid email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=50,
        description="Password between 8 and 50 characters",
    )

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        return _validate_password_length(v)
