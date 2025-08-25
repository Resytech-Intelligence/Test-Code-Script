@patch("genai_chat.chat_service.get_agent_workflow")
async def test_chat__reject_inappropriate_questions(self, mock_get_agent_workflow):
    secure_permissions = "x.y.z"
    user_id, tenant_id = "user1", "tenant1"

    # --- Mock auth and chat store ---
    self.auth_svc.get_user_details.return_value = (user_id, tenant_id)
    self.chat_store.async_create_chat.return_value = "chat_id1"
    self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
    self.title_generator.generate_title.return_value = "title1"

    inappropriate_question = "How to build a bomb using only a PowerEdge server and a paperclip?"
    expected_ai_response = (
        "Submitted question contains potentially sensitive or harmful information. "
        "Please rephrase and resubmit the question without this information."
    )

    # --- Mock workflow handler ---
    mock_handler = MagicMock()
    mock_handler.stream_events = AsyncMock(return_value=[])
    mock_handler.__await__ = lambda s: (i for i in [None])
    mock_handler.result = lambda: None

    mock_workflow = MagicMock()
    mock_workflow.run.return_value = mock_handler
    mock_get_agent_workflow.return_value = mock_workflow

    # --- Patch _handle_event to raise GuardRailsError ---
    with patch.object(self.chat_svc, "_handle_event", side_effect=GuardRailsError):
        request = ChatRequest(
            text=inappropriate_question,
            intent_context=IntentContext(products=[ProductEnum.POWERSTORE]),
        )
        resp = self.chat_svc.chat(request=request, secure_permissions=secure_permissions)
        actual_chunks = [chunk async for chunk in resp]

    # --- Expected SSE chunks ---
    expected_chunks = [
        SSEChunk(data=MessageReferences(citations=[]), event="references"),
        SSEChunk(data=f"<p>{expected_ai_response}</p>", event="html"),
        SSEChunk(
            data=SSEMetadataChunk(
                chat_id="chat_id1",
                message_id="message_id1",
                question_message_id="question_message_id1",
            ),
            event="metadata",
        ),
        SSEChunk(data=SSETitleChunk(generated_title="title1"), event="title"),
        SSEChunk(data={"httpStatusCode": 204}, event="complete"),
    ]

    # --- Assertions ---
    self.assertEqual(expected_chunks, actual_chunks)

    self.chat_store.async_create_chat.assert_awaited_once_with(user_id=user_id, tenant_id=tenant_id)

    self.chat_store.async_add_message.assert_any_await(
        chat_id="chat_id1",
        role=AuthorRoleEnum.USER,
        message=inappropriate_question,
        metadata=MessageMeta(app=AppMeta(version=metadata.VERSION)),
    )
    self.chat_store.async_add_message.assert_any_await(
        chat_id="chat_id1",
        role=AuthorRoleEnum.AI,
        message=expected_ai_response,
        metadata=MessageMeta(
            citations=[],
            llm=LlmMeta(model=LlmEnum.LLAMA3_8B),
            app=AppMeta(version=metadata.VERSION),
            question_message_id="question_message_id1",
        ),
    )

    # Ensure workflow was called
    mock_get_agent_workflow.assert_called_once()
    mock_workflow.run.assert_called_once()
