"""User service layer - business logic for user operations."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_password_hash, verify_password
from app.db.models.user import User
from app.strings import get_message


def auth_service_factory_provider() -> Callable[[AsyncSession], AuthService]:
    """Return a factory for session-scoped AuthService."""

    def factory(session: AsyncSession) -> AuthService:
        return AuthService(session)

    return factory


class AuthenticationError(Exception):
    """Base error for authentication-related failures."""

    def __init__(self, error_code: str) -> None:
        super().__init__(get_message(f"auth.{error_code}"))
        self.error_code = error_code


class UserAlreadyExistsError(AuthenticationError):
    """Raised when attempting to register a user that already exists."""

    def __init__(self) -> None:
        super().__init__("user_exists")


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""

    def __init__(self) -> None:
        super().__init__("invalid_credentials")


class PasswordTooLongError(AuthenticationError):
    """Raised when password exceeds the maximum allowed length (72 bytes)."""

    def __init__(self) -> None:
        super().__init__("password_too_long")


class AuthService:
    """Service for user registration, authentication, and deletion."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register_user(self, email: str, password: str) -> User:
        """Register a new user.

        Args:
            email: User's email address
            password: Plain text password (will be hashed)

        Returns:
            The newly created User object

        Raises:
            UserAlreadyExistsError: If email is already registered
            PasswordTooLongError: If password exceeds 72 bytes when UTF-8 encoded
        """
        result = await self._session.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise UserAlreadyExistsError()

        try:
            hashed_password = get_password_hash(password)
        except ValueError as e:
            raise PasswordTooLongError() from e

        try:
            result = await self._session.execute(
                insert(User).values(email=email, hashed_password=hashed_password).returning(User)
            )
            new_user = result.scalar_one()
        except IntegrityError as e:
            error_str = str(e.orig).lower()
            if (
                "unique" in error_str
                or "duplicate" in error_str
                or "23505" in str(e.orig)
                or "ix_users_email" in error_str
            ):
                raise UserAlreadyExistsError() from e
            raise

        return new_user

    async def authenticate_user(self, email: str, password: str) -> User:
        """Authenticate a user with email and password.

        Args:
            email: User's email address
            password: Plain text password

        Returns:
            The authenticated User object

        Raises:
            InvalidCredentialsError: If email or password is incorrect
            PasswordTooLongError: If password exceeds 72 bytes when UTF-8 encoded
        """
        result = await self._session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user is None:
            raise InvalidCredentialsError()

        try:
            password_valid = verify_password(password, user.hashed_password)
        except ValueError as e:
            raise PasswordTooLongError() from e

        if not password_valid:
            raise InvalidCredentialsError()

        return user

    async def delete_user(self, user_id: int) -> int:
        """Delete a user and all associated data.

        Args:
            user_id: ID of the user to delete

        Returns:
            The ID of the deleted user

        Note:
            This performs a hard delete. Associated data is removed via CASCADE.
        """
        await self._session.execute(delete(User).where(User.id == user_id))
        return user_id

    async def get_user_by_id(self, user_id: int) -> User | None:
        """Return the user with the given ID, or None if not found."""
        result = await self._session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
