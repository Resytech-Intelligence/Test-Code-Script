@patch("genai_chat.validation.guard_rails.LLMGuardRails.async_validate_user_question")
async def test_astream_chat__uses_guard_rails(self, mock_async_validate_user_question):
    # Arrange
    question = "Does PowerStore store electricity?"
    secure_permissions = "mock_token"
    chat_request = ChatRequest(text=question)

    # Act
    response_generator = self.chat_service.chat(
        request=chat_request,
        secure_permissions=secure_permissions
    )

    # The chat service returns an async generator. We need to exhaust it
    # to ensure all logic (including the guard rails call) is executed.
    async for chunk in response_generator:
        # We don't care about the chunks here, just that the
        # generator runs to completion.
        pass

    # Assert
    # The core assertion remains the same as the original test
    mock_async_validate_user_question.assert_awaited_with(question=question)
