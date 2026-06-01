"""Interest extraction service - business logic for extracting interests from prompts."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.client import LLMClient, LLMServiceError
from app.llm.schemas import InterestExtractionResult
from app.strings import get_message


def interest_service_factory_provider(
    llm_client: LLMClient,
) -> Callable[[AsyncSession], InterestService]:
    """Return a factory for session-scoped InterestService (closes over the given LLM client)."""

    def factory(session: AsyncSession) -> InterestService:
        return InterestService(session, llm_client)

    return factory


class InterestExtractionError(Exception):
    """Base error for interest extraction failures."""

    def __init__(self, error_code: str) -> None:
        super().__init__(get_message(f"interests.{error_code}"))
        self.error_code = error_code


class InterestService:
    """Service for extracting interests from natural language prompts via LLM."""

    def __init__(self, session: AsyncSession, llm_client: LLMClient) -> None:
        self._session = session
        self._llm_client = llm_client

    async def extract_interests_from_prompt(self, prompt: str) -> InterestExtractionResult:
        """Extract interests from a natural language prompt using LLM.

        Returns:
            InterestExtractionResult with add_interests and remove_interests.

        Raises:
            InterestExtractionError: When the LLM layer fails (unavailable, auth, invalid response).
        """
        try:
            return await self._llm_client.extract_interests(prompt)
        except LLMServiceError as exc:
            raise InterestExtractionError(exc.error_code) from exc
