# Copyright © 2016-2025 Dell Inc. or its subsidiaries.
# All Rights Reserved.
# ruff: noqa: E402, I001
import os
from tqdm.contrib.itertools import product
from workflows import Context
from genai_chat.llms.triton import Triton
from genai_chat.tools import UserContext
from genai_chat.workflow.agent_workflow import get_agent_workflow
os.environ["PROFILE"] = "unittest"
from unittest import skip
from llama_index.core.base.llms.types import ChatMessage as LlamaChatMessage
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.llms.types import ChatMessage as LiChatMessage
from genai_chat.auth import AuthService
from genai_chat.config import conf, mtls
from genai_chat.storage.chat_store.postgres import ChatRepository
from resources import metadata
import unittest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from genai_chat.chat_service import ChatService, TitleGenerator
from genai_chat.domain.api_model import (
  Chat,
  ChatUpdate,
  MessageFeedback,
  ChatMessageResponse,
  Author,
  PaginatedMessagesResponse,
  ChatRequest,
  SSEChunk,
  MessageReferences,
  SSEMetadataChunk,
  SSETitleChunk,
  MessageMeta,
  AppMeta,
  LlmMeta,
  ChatMessage,
  Citation,
  IntentContext,
  AliasedProductEnum, ModelHyperParams,
)
from genai_chat.domain.enums import (
  MsgFeedbackRatingEnum,
  MsgFeedbackCategoryEnum,
  AuthorRoleEnum,
  SensitiveDataTypesEnum,
  LlmEnum,
  ProductEnum,
  HitlQuestionTypeEnum,
)
from genai_chat.domain.errors import UnauthorizedError, SensitiveDataError, GuardRailsError
class TestChatService(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.chat_store = MagicMock(spec=ChatRepository)
    self.auth_svc = MagicMock(spec=AuthService)
    self.title_generator = MagicMock(spec=TitleGenerator)
    self.chat_svc = ChatService(
      chat_store=self.chat_store,
      auth_service=self.auth_svc,
      title_generator=self.title_generator,
    )
  async def test_get_chats(self):
    expected = [Chat(chat_id="chat_id1", created_at=datetime.now()), Chat(chat_id="chat_id2", created_at=datetime.now())]
    secure_permissions = "x.y.z"
    self.chat_store.async_get_chats.return_value = expected
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    actual = await self.chat_svc.get_chats(secure_permissions=secure_permissions)
    self.assertEqual(expected, actual)
    self.chat_store.async_get_chats.assert_awaited_once_with(user_id="user1", tenant_id="tenant1")
  async def test_get_chat(self):
    expected = [Chat(chat_id="chat_id1", created_at=datetime.now())]
    secure_permissions = "x.y.z"
    self.chat_store.async_get_chat.return_value = expected
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    actual = await self.chat_svc.get_chat(chat_id="chat_id1", secure_permissions=secure_permissions)
    self.assertEqual(expected, actual)
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id="chat_id1")
    self.chat_store.async_get_chat.assert_awaited_once_with(chat_id="chat_id1")
  async def test_update_chat(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    request = ChatUpdate(title="new title")
    await self.chat_svc.update_chat(chat_id=chat_id, chat=request, secure_permissions=secure_permissions)
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id=chat_id)
    self.chat_store.async_rename_chat.assert_awaited_once_with(chat_id, request.title)
  async def test_update_chat__unauthorized_access_to_chat_id(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    request = ChatUpdate(title="new title")
    self.auth_svc.async_validate_chat_id.side_effect = UnauthorizedError()
    with self.assertRaises(UnauthorizedError):
      await self.chat_svc.update_chat(chat_id=chat_id, chat=request, secure_permissions=secure_permissions)
    self.chat_store.async_rename_chat.assert_not_awaited()
  async def test_delete_chat(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    await self.chat_svc.delete_chat(chat_id=chat_id, secure_permissions=secure_permissions)
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id=chat_id)
    self.chat_store.async_soft_delete_chat.assert_awaited_once_with(chat_id)
  async def test_delete_chat__unauthorized_access_to_chat_id(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    self.auth_svc.async_validate_chat_id.side_effect = UnauthorizedError()
    with self.assertRaises(UnauthorizedError):
      await self.chat_svc.delete_chat(chat_id=chat_id, secure_permissions=secure_permissions)
    self.chat_store.async_soft_delete_chat.assert_not_awaited()
  async def test_add_feedback(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    message_id = "msg1"
    feedback = MessageFeedback(
      rating=MsgFeedbackRatingEnum.THUMBS_UP, categories=[MsgFeedbackCategoryEnum.OTHER], text="comment"
    )
    await self.chat_svc.add_feedback(
      chat_id=chat_id, message_id=message_id, feedback=feedback, secure_permissions=secure_permissions
    )
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id=chat_id)
    self.chat_store.async_upsert_feedback.assert_awaited_once_with(message_id=message_id, feedback=feedback)
  async def test_add_feedback__unauthorized_access_to_chat_id(self):
    secure_permissions = "x.y.z"
    chat_id = "chat1"
    message_id = "msg1"
    feedback = MessageFeedback(
      rating=MsgFeedbackRatingEnum.THUMBS_UP, categories=[MsgFeedbackCategoryEnum.OTHER], text="comment"
    )
    self.auth_svc.async_validate_chat_id.side_effect = UnauthorizedError()
    with self.assertRaises(UnauthorizedError):
      await self.chat_svc.add_feedback(
        chat_id=chat_id, message_id=message_id, feedback=feedback, secure_permissions=secure_permissions
      )
    self.chat_store.async_upsert_feedback.assert_not_awaited()
  async def test_get_messages(self):
    created_at = datetime.now()
    messages = [
      ChatMessage(
        chat_id="chat_id1",
        message_id="msg1",
        created_at=created_at,
        author=Author(role=AuthorRoleEnum.AI),
        text="foo bar",
        metadata=MessageMeta(),
      )
    ]
    messages_response = [
      ChatMessageResponse(
        chat_id="chat_id1",
        message_id="msg1",
        created_at=created_at,
        author=Author(role=AuthorRoleEnum.AI),
        text="<p>foo bar</p>",
        layouts=[],
      )
    ]
    secure_permissions = "x.y.z"
    self.chat_store.async_get_total_message_count.return_value = 13
    self.chat_store.async_get_chat_messages.return_value = messages
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    actual = await self.chat_svc.get_messages(chat_id="chat_id1", page=None, per_page=2, secure_permissions=secure_permissions)
    expected = PaginatedMessagesResponse(
      messages=messages_response,
      metadata=PaginatedMessagesResponse.Meta(page=7, total_message_count=13),
    )
    self.assertEqual(expected, actual)
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id="chat_id1")
    self.chat_store.async_get_chat_messages.assert_awaited_once_with(chat_id="chat_id1", limit=2, offset=12, order="asc")
  async def test_get_chat_messages_by_id(self):
    time = datetime.now()
    messages = [
      ChatMessage(
        chat_id="chat_id1",
        message_id="msg1",
        created_at=time,
        author=Author(role=AuthorRoleEnum.USER),
        text="foo bar",
        metadata=MessageMeta(),
      )
    ]
    secure_permissions = "x.y.z"
    self.chat_store.async_get_chat_messages_by_id.return_value = messages
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    actual = await self.chat_svc.get_messages_by_id(
      chat_id="chat_id1", message_ids=["msg1"], secure_permissions=secure_permissions
    )
    expected_messages = [
      ChatMessageResponse(
        chat_id="chat_id1",
        message_id="msg1",
        created_at=time,
        author=Author(role=AuthorRoleEnum.USER),
        text="foo bar",
        layouts=[],
      )
    ]
    expected = PaginatedMessagesResponse(
      messages=expected_messages,
      metadata=PaginatedMessagesResponse.Meta(page=1, total_message_count=1),
    )
    self.assertEqual(expected, actual)
    self.auth_svc.async_validate_chat_id.assert_awaited_once_with(secure_permissions=secure_permissions, chat_id="chat_id1")
    self.chat_store.async_get_chat_messages_by_id.assert_awaited_once_with(chat_id="chat_id1", message_ids=["msg1"])
  # @skip("TODO: rewrite to work with AgentWorkflow instead of ChatEngine")
  # @patch("genai_chat.chat_service.get_chat_engine")
  # async def test_chat(self, get_chat_engine):
  #   secure_permissions = "x.y.z"
  #   mock_chat_engine = MagicMock()
  #   get_chat_engine.return_value = mock_chat_engine
  #   self.setup_mock_astream_chat_response(
  #     mock_chat_engine,
  #     ["foo", " ", "bar"],
  #     source_nodes=[
  #       NodeWithScore(
  #         node=TextNode(
  #           text="blah blah",
  #           metadata={
  #             "doc_datasource": "INFO_HUB",
  #             "title": "Hello-World!",
  #             "filename": "abc.md",
  #             "modified_date": 0,
  #             "link": "a",
  #           },
  #         ),
  #         score=0.1,
  #       ),
  #       NodeWithScore(
  #         node=TextNode(
  #           text="blah blah",
  #           metadata={"doc_datasource": "INFO_HUB", "filename": "h18013.md", "modified_date": 0, "link": "b"},
  #         ),
  #         score=0.1,
  #       ),
  #       NodeWithScore(
  #         node=TextNode(
  #           text="blah blah",
  #           metadata={
  #             "doc_datasource": "INFO_HUB",
  #             "filename": "h19723-poweredge-mx-advanced-npar.md",
  #             "modified_date": 0,
  #             "link": "c",
  #           },
  #         ),
  #         score=0.1,
  #       ),
  #       NodeWithScore(
  #         node=TextNode(
  #           text="blah blah",
  #           metadata={
  #             "doc_datasource": "INFO_HUB",
  #             "filename": "h19723_poweredge-is-the_best.md",
  #             "modified_date": 0,
  #             "link": "d",
  #           },
  #         ),
  #         score=0.1,
  #       ),
  #       NodeWithScore(
  #         node=TextNode(
  #           text="blah blah",
  #           metadata={
  #             "doc_datasource": "INFO_HUB",
  #             "filename": "H20235%20-%20PowerEdge%2015G%20and%2016G%20Connectivity%20and%20Open%20Telemetry.md",
  #             "creation_date": 1,
  #             "link": "d",
  #           },
  #         ),
  #         score=0.2,
  #       ),
  #     ],
  #   )
  #   self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
  #   self.chat_store.async_create_chat.return_value = "chat_id1"
  #   self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
  #   self.title_generator.generate_title.return_value = "What?"
  #   resp = self.chat_svc.chat(request=ChatRequest(text="What is PowerStore?"), secure_permissions=secure_permissions)
  #   actual_chunks = [chunk async for chunk in resp]
  #   expected_chunks = [
  #     SSEChunk(
  #       data=MessageReferences(
  #         citations=[
  #           Citation(title="Hello-World!", link="a", published_date=0, score=0.1),
  #           Citation(title="H18013", link="b", published_date=0, score=0.1),
  #           Citation(title="Poweredge Mx Advanced Npar", link="c", published_date=0, score=0.1),
  #           Citation(title="Poweredge Is the Best", link="d", published_date=0, score=0.1),
  #           Citation(
  #             title="PowerEdge 15G and 16G Connectivity and Open Telemetry", link="d", published_date=1, score=0.2
  #           ),
  #         ]
  #       ),
  #       event="references",
  #     ),
  #     SSEChunk(data="<p>foo </p>", event="html"),
  #     SSEChunk(data="<p>foo bar</p>", event="html"),
  #     SSEChunk(
  #       data=SSEMetadataChunk(chat_id="chat_id1", message_id="message_id1", question_message_id="question_message_id1"),
  #       event="metadata",
  #     ),
  #     SSEChunk(data=SSETitleChunk(generated_title="What?"), event="title"),
  #     SSEChunk(data={"httpStatusCode": 204}, event="complete"),
  #   ]
  #   self.assertEqual(expected_chunks, actual_chunks)
  #   self.auth_svc.get_user_details.assert_called_with(secure_permissions=secure_permissions)
  #   self.chat_store.async_create_chat.assert_awaited_once_with(user_id="user1", tenant_id="tenant1")
  #   self.chat_store.async_add_message.assert_awaited()



  @patch("genai_chat.chat_service.get_agent_workflow")
  async def test_chat(self, mock_get_agent_workflow):
      secure_permissions = "x.y.z"
      user_id, tenant_id = "user1", "tenant1"

      # --- Mock AgentWorkflow and its handler ---
      mock_workflow = MagicMock()
      mock_handler = MagicMock()
      mock_handler.stream_events = AsyncMock()

      # Fake events that handler.stream_events() will yield
      async def fake_events():
        yield MagicMock(data="foo")
        yield MagicMock(data="bar")

      mock_handler.stream_events.side_effect = fake_events
      mock_handler.result.return_value = None
      mock_workflow.run.return_value = mock_handler
      mock_get_agent_workflow.return_value = mock_workflow


      self.chat_svc._handle_event = AsyncMock(
        side_effect=lambda event, resp_ctx, handler: [
            SSEChunk(data=f"<p>{event.data}</p>", event="html")
        ]
      )

      # --- Mock services used inside chat() ---
      self.auth_svc.get_user_details.return_value = (user_id, tenant_id)
      self.chat_store.async_create_chat.return_value = "chat_id1"
      self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
      self.title_generator.generate_title.return_value = "What?"

      # --- Prepare the ChatRequest ---
      request = ChatRequest(text="What is PowerStore?")

      # --- Call the chat function ---
      resp = self.chat_svc.chat(request=request, secure_permissions=secure_permissions)
      actual_chunks = [chunk async for chunk in resp]

      # --- Expected SSE Chunks ---
      expected_chunks = [
          # Simulated HTML chunks from fake events (converted in _handle_event)
          SSEChunk(data="<p>foo</p>", event="html"),
          SSEChunk(data="<p>bar</p>", event="html"),
          # Metadata chunk
          SSEChunk(
              event="metadata",
              data=SSEMetadataChunk(
                  chat_id="chat_id1",
                  message_id="message_id1",
                  question_message_id="question_message_id1",
              ),
          ),
          # Title chunk for new conversation
          SSEChunk(data=SSETitleChunk(generated_title="What?"), event="title"),
          # Complete chunk
          SSEChunk(data={"httpStatusCode": 204}, event="complete"),
      ]

      # --- Assertions ---
      self.assertEqual(expected_chunks, actual_chunks)
      self.auth_svc.get_user_details.assert_called_with(secure_permissions=secure_permissions)
      self.chat_store.async_create_chat.assert_awaited_once_with(user_id=user_id, tenant_id=tenant_id)
      self.chat_store.async_add_message.assert_awaited()
      mock_get_agent_workflow.assert_called_once()
      mock_workflow.run.assert_called_once()  # Ensure workflow.run was called
      mock_handler.stream_events.assert_awaited()  # Ensure stream_events was awaited

  # @skip("TODO: rewrite to work with AgentWorkflow instead of ChatEngine")
  # @patch("genai_chat.chat_service.get_chat_engine")
  # async def test_chat__considers_chat_history(self, get_chat_engine):
  #   secure_permissions = "x.y.z"
  #   mock_chat_engine = MagicMock()
  #   get_chat_engine.return_value = mock_chat_engine
  #   self.setup_mock_astream_chat_response(mock_chat_engine, ["foo", "bar"])
  #   self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
  #   self.chat_store.async_add_message.return_value = "message_id1"
  #   msg_history = [
  #     ChatMessage(
  #       chat_id="chat_id1",
  #       message_id="msg0",
  #       created_at=datetime.now(),
  #       author=Author(role=AuthorRoleEnum.AI),
  #       text="old content",
  #       metadata=MessageMeta(),
  #     )
  #   ]
  #   self.chat_store.async_get_chat_messages.return_value = msg_history
  #   async for _ in self.chat_svc.chat(
  #     request=ChatRequest(
  #       chat_id="chat_id1", text="What is PowerStore?", intent_context=IntentContext(products=[ProductEnum.POWERSTORE])
  #     ),
  #     secure_permissions=secure_permissions,
  #   ):
  #     ...
  #   mock_chat_engine.astream_chat.assert_awaited_once_with(
  #     "What is PowerStore?",
  #     [
  #       LlamaChatMessage(
  #         role=MessageRole.ASSISTANT,
  #         content="old content",
  #       )
  #     ],
  #   )
  

  
  @patch("genai_chat.chat_service.get_agent_workflow")
  async def test_chat__considers_chat_history(self, mock_get_agent_workflow):
    secure_permissions = "x.y.z"

    # --- Mock auth and chat store ---
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    self.chat_store.async_add_message.return_value = "message_id1"

    # Chat history contains previous AI message
    msg_history = [
        ChatMessage(
            chat_id="chat_id1",
            message_id="msg0",
            created_at=None,
            author=Author(role=AuthorRoleEnum.AI),
            text="old content",
            metadata=MessageMeta(),
        )
    ]
    self.chat_store.async_get_chat_messages.return_value = msg_history

    # --- Mock workflow handler ---
    async def fake_stream_events():
        yield SSEChunk(data="<p>foo</p>", event="html")
        yield SSEChunk(data="<p>bar</p>", event="html")

    mock_handler = MagicMock()
    mock_handler.stream_events = fake_stream_events
    mock_handler.__await__ = lambda s: (i for i in [None])
    mock_handler.result = lambda: None  # simulate no exception

    # --- Mock workflow returned by get_agent_workflow ---
    mock_workflow = MagicMock()
    mock_workflow.run.return_value = mock_handler
    mock_get_agent_workflow.return_value = mock_workflow

    # --- Chat request ---
    request = ChatRequest(
        chat_id="chat_id1",
        text="What is PowerStore?",
        intent_context=IntentContext(products=[ProductEnum.POWERSTORE]),
    )

    # --- Call the chat() method ---
    async for _ in self.chat_svc.chat(request=request, secure_permissions=secure_permissions):
        pass  # just consume output

    # --- Assertions ---
    # Check that chat history was passed to workflow.run
    args, kwargs = mock_workflow.run.call_args
    assert "chat_history" in kwargs
    chat_history_passed = kwargs["chat_history"]

    # Ensure it's a list of LiChatMessage with correct content
    assert all(isinstance(msg, LiChatMessage) for msg in chat_history_passed)
    assert chat_history_passed[0].content == "old content"
    assert chat_history_passed[0].role == "assistant"  # AI message maps to 'assistant'

    # Ensure stream_events was called
    mock_handler.stream_events.assert_called()

    # Ensure chat messages were fetched
    self.chat_store.async_get_chat_messages.assert_awaited_once_with(
        chat_id="chat_id1", limit=20, offset=0, order="desc"
    )



  async def test_chat__checks_question_for_sensitive_content(self):
    secure_permissions = "x.y.z"
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    with self.assertRaises(SensitiveDataError):
      async for _ in self.chat_svc.chat(
        request=ChatRequest(
          text="My social security number is 555-55-5555", intent_context=IntentContext(products=[ProductEnum.POWERSTORE])
        ),
        secure_permissions=secure_permissions,
      ):
        ...
    self.chat_store.async_add_rejected_message.assert_awaited_once_with(
      chat_id=None,
      message="My social security number is [SSN]",
      user_id="user1",
      tenant_id="tenant1",
      rejected_reason=[SensitiveDataTypesEnum.SSN],
    )
    self.chat_store.async_create_chat.assert_not_awaited()
    self.chat_store.async_add_message.assert_not_awaited()
  """
  #@skip("TODO: rewrite to work with AgentWorkflow instead of ChatEngine")
  @patch("genai_chat.workflow.agent_workflow.get_agent_workflow")
  async def test_chat__sanitize_user_input(self, get_agent_workflow):
    mock_chat_engine = MagicMock()
    get_agent_workflow.return_value = mock_chat_engine
    self.setup_mock_astream_chat_response(mock_chat_engine, ["foo", "bar"])
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    self.chat_store.async_create_chat.return_value = "chat_id1"
    self.chat_store.async_add_message.return_value = "message_id1"
    self.title_generator.generate_title.return_value = "title1"
    raw_question = "\n What\n is \n PowerStore? \n\n"
    sanitized_question = "What\n is \n PowerStore?"
    async for _ in self.chat_svc.chat(request=ChatRequest(text=raw_question), secure_permissions="x.y.z"):
      ...
    mock_chat_engine.astream_chat.assert_awaited_once_with(sanitized_question, [])
    self.chat_store.async_add_message.assert_any_await(
      chat_id="chat_id1",
      role=AuthorRoleEnum.USER,
      message=raw_question,
      metadata=MessageMeta(app=AppMeta(version=metadata.VERSION)),
    )
    """
  @patch("genai_chat.chat_service.get_agent_workflow")
  async def test_chat_sanitize_user_input(self, get_agent_workflow):
    mock_agent_workflow = MagicMock()
    get_agent_workflow.return_value = mock_agent_workflow
    mock_agent_workflow.run.return_value = {"text": "What is in PowerStore?"}
    self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
    self.chat_store.async_create_chat.return_value = "chat_id1"
    self.chat_store.async_add_message.return_value = "message_id1"
    self.title_generator.generate_title.return_value = "title1"
    raw_question = "\n What\n is\n in PowerStore? \n\n"
    sanitized_question = "What is in PowerStore?"
    request = ChatRequest(
          text=raw_question,
          intent_context=IntentContext(products=[ProductEnum.POWERSTORE])
        ),
    secure_permissions="x.y.z"
    async for _ in self.chat_svc.chat(request):
      pass
    mock_agent_workflow.run.assert_called_once_with(
      user_input=sanitized_question,
      context={"user": "user1", "tenant": "tenant1"}
    )
    self.chat_store.async_add_message.assert_any_await(
      chat_id="chat_id1",
      role=AuthorRoleEnum.USER,
      message=raw_question,
      metadata=MessageMeta(app=AppMeta(version=metadata.VERSION))
    )
  # @skip("TODO: rewrite to work with AgentWorkflow instead of ChatEngine")
  # @patch("genai_chat.chat_service.get_chat_engine")
  # async def test_chat__reject_inappropriate_questions(self, get_chat_engine):
  #   mock_chat_engine = MagicMock()
  #   mock_chat_engine.astream_chat = AsyncMock()
  #   mock_chat_engine.astream_chat.side_effect = GuardRailsError()
  #   get_chat_engine.return_value = mock_chat_engine
  #   self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
  #   self.chat_store.async_create_chat.return_value = "chat_id1"
  #   self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
  #   self.title_generator.generate_title.return_value = "title1"
  #   inappropriate_question = "How to build a bomb using only a PowerEdge server and a paperclip?"
  #   expected_ai_response = "Submitted question contains potentially sensitive or harmful information. Please rephrase and resubmit the question without this information."
  #   resp = self.chat_svc.chat(
  #     request=ChatRequest(text=inappropriate_question, intent_context=IntentContext(products=[ProductEnum.POWERSTORE])),
  #     secure_permissions="x.y.z",
  #   )
  #   actual_chunks = [chunk async for chunk in resp]
  #   expected_chunks = [
  #     SSEChunk(data=MessageReferences(citations=[]), event="references"),
  #     SSEChunk(data=f"<p>{expected_ai_response}</p>", event="html"),
  #     SSEChunk(
  #       data=SSEMetadataChunk(chat_id="chat_id1", message_id="message_id1", question_message_id="question_message_id1"),
  #       event="metadata",
  #     ),
  #     SSEChunk(data=SSETitleChunk(generated_title="title1"), event="title"),
  #     SSEChunk(data={"httpStatusCode": 204}, event="complete"),
  #   ]
  #   self.assertEqual(expected_chunks, actual_chunks)
  #   self.chat_store.async_create_chat.assert_awaited_once_with(user_id="user1", tenant_id="tenant1")
  #   self.chat_store.async_add_message.assert_any_await(
  #     chat_id="chat_id1",
  #     role=AuthorRoleEnum.USER,
  #     message=inappropriate_question,
  #     metadata=MessageMeta(app=AppMeta(version=metadata.VERSION)),
  #   )
  #   self.chat_store.async_add_message.assert_any_await(
  #     chat_id="chat_id1",
  #     role=AuthorRoleEnum.AI,
  #     message=expected_ai_response,
  #     metadata=MessageMeta(
  #       citations=[],
  #       llm=LlmMeta(model=LlmEnum.LLAMA3_8B),
  #       app=AppMeta(version=metadata.VERSION),
  #       question_message_id="question_message_id1",
  #     ),
  #   )




  
  @patch("genai_chat.chat_service.get_agent_workflow")
  async def test_chat__reject_inappropriate_questions(self, mock_get_agent_workflow):
      secure_permissions = "x.y.z"
      user_id, tenant_id = "user1", "tenant1"

      # --- Mock auth and chat_store ---
      self.auth_svc.get_user_details.return_value = (user_id, tenant_id)
      self.chat_store.async_create_chat.return_value = "chat_id1"
      self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
      self.title_generator.generate_title.return_value = "title1"

      inappropriate_question = "How to build a bomb using only a PowerEdge server and a paperclip?"
      expected_ai_response = (
          "Submitted question contains potentially sensitive or harmful information. "
          "Please rephrase and resubmit the question without this information."
      )

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


  # @skip("No longer rejecting questions without product in the text. TODO: remove this test")
  # @patch("genai_chat.chat_service.get_chat_engine")
  # async def test_chat__reject_questions_without_product(self, get_chat_engine):
  #   mock_chat_engine = MagicMock()
  #   mock_chat_engine.astream_chat = AsyncMock()
  #   # mock_chat_engine.astream_chat.side_effect = MissingProductError()
  #   get_chat_engine.return_value = mock_chat_engine
  #   self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
  #   self.chat_store.async_create_chat.return_value = "chat_id1"
  #   self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
  #   self.title_generator.generate_title.return_value = "title1"
  #   bad_question = "How?"
  #   expected_ai_response = 'To provide the best answer to your question, please provide the product name of your system. An example of a product name is "PowerStore."'
  #   resp = self.chat_svc.chat(request=ChatRequest(text=bad_question), secure_permissions="x.y.z")
  #   actual_chunks = [chunk async for chunk in resp]
  #   expected_chunks = [
  #     SSEChunk(data=MessageReferences(citations=[]), event="references"),
  #     SSEChunk(data=f"<p>{expected_ai_response}</p>", event="html"),
  #     SSEChunk(
  #       data=SSEMetadataChunk(chat_id="chat_id1", message_id="message_id1", question_message_id="question_message_id1"),
  #       event="metadata",
  #     ),
  #     SSEChunk(data=SSETitleChunk(generated_title="title1"), event="title"),
  #     SSEChunk(data={"httpStatusCode": 204}, event="complete"),
  #   ]
  #   self.assertEqual(expected_chunks, actual_chunks)
  #   self.chat_store.async_create_chat.assert_awaited_once_with(user_id="user1", tenant_id="tenant1")
  #   self.chat_store.async_add_message.assert_any_await(
  #     chat_id="chat_id1",
  #     role=AuthorRoleEnum.USER,
  #     message=bad_question,
  #     metadata=MessageMeta(app=AppMeta(version=metadata.VERSION)),
  #   )
  #   self.chat_store.async_add_message.assert_any_await(
  #     chat_id="chat_id1",
  #     role=AuthorRoleEnum.AI,
  #     message=expected_ai_response,
  #     metadata=MessageMeta(
  #       citations=[],
  #       llm=LlmMeta(model=LlmEnum.LLAMA3_8B),
  #       app=AppMeta(version=metadata.VERSION),
  #       question_message_id="question_message_id1",
  #     ),
  #   )

  @patch("genai_chat.chat_service.get_agent_workflow")
  async def test_chat__reject_questions_without_product(self, mock_get_agent_workflow):
      secure_permissions = "x.y.z"
      self.auth_svc.get_user_details.return_value = ("user1", "tenant1")
      self.chat_store.async_create_chat.return_value = "chat_id1"
      self.chat_store.async_add_message.side_effect = ["question_message_id1", "message_id1"]
      self.title_generator.generate_title.return_value = "title1"

      mock_workflow = MagicMock()
      mock_handler = MagicMock()
      mock_handler.stream_events = AsyncMock()
      mock_workflow.run.return_value = mock_handler
      mock_get_agent_workflow.return_value = mock_workflow

      bad_question = "How?"
      expected_ai_response = 'To provide the best answer to your question, please provide the product name of your system. An example of a product name is "PowerStore."'

      resp = self.chat_svc.chat(request=ChatRequest(text=bad_question), secure_permissions=secure_permissions)
      actual_chunks = [chunk async for chunk in resp]

      expected_chunks = [
          SSEChunk(data=MessageReferences(citations=[]), event="references"),
          SSEChunk(data=f"<p>{expected_ai_response}</p>", event="html"),
          SSEChunk(data=SSEMetadataChunk(chat_id="chat_id1", message_id="message_id1", question_message_id="question_message_id1"), event="metadata"),
          SSEChunk(data=SSETitleChunk(generated_title="title1"), event="title"),
          SSEChunk(data={"httpStatusCode": 204}, event="complete"),
      ]

      self.assertEqual(expected_chunks, actual_chunks)
      self.chat_store.async_create_chat.assert_awaited_once_with(user_id="user1", tenant_id="tenant1")
      self.chat_store.async_add_message.assert_any_await(
          chat_id="chat_id1",
          role=AuthorRoleEnum.USER,
          message=bad_question,
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

  def setup_mock_astream_chat_response(
    self, mock_chat_engine: MagicMock, text_chunks: list[str], source_nodes: list[NodeWithScore] | None = None
  ) -> None:
    mock_chat_engine.astream_chat = AsyncMock()
    if source_nodes is None:
      source_nodes = [
        NodeWithScore(
          node=TextNode(text="blah blah"),
          score=0.1,
        )
      ]
    async def response_stream_from_llm():
      for chunk in text_chunks:
        yield ChatResponse(message=LlamaChatMessage(text=chunk), delta=chunk)
    # TODO: we no longer use StreamingAgentChatResponse
    # mock_chat_engine.astream_chat.return_value = CustomStreamingAgentChatResponse(
    # achat_stream=response_stream_from_llm(),
    # source_nodes=source_nodes,
    # )
  @patch("genai_chat.tools.util.LlmHumanInTheLoopGenericResponseExtractor.validate_and_extract_user_response")
  async def test__validate_HITL_user_response(self, validate_and_extract_user_response):
    llm = Triton(
      model_name="Meta_Llama_3_8B_Instruct",
      base_url=conf.models.llm_llama3_base_url,
      ssl_context=mtls.get_ssl_context(),
      temperature=0,
      is_chat_model=True,
    )
    user_context = UserContext(
      user_id="user_id",
      tenant_id="tenant_id",
      secure_permissions="secure_permissions",
      entitlements=set(),
      intent=IntentContext(tools=set(), products=[AliasedProductEnum.UNITY], objects=[]),
    )
    workflow = get_agent_workflow(llm=llm, user_context=user_context, products=[])
    ctx: Context = Context(workflow, stepwise=False)
    await ctx.store.set("hitl_follow_up_question", "Would you like to know about iops, latency, or storage")
    await ctx.store.set(
      "hitl_validate_user_response", {"validate_response": True, "question_type": HitlQuestionTypeEnum.MULTIPLE_CHOICE}
    )
    validate_and_extract_user_response.return_value = True, "latency"
    actual_user_response_valid, actual_extracted_user_response = await self.chat_svc._validate_hitl_user_response(
      ctx, "I want to know about latency"
    )
    validate_and_extract_user_response.assert_called()
    self.assertTrue(actual_user_response_valid)
    self.assertEqual(actual_extracted_user_response, "latency")
    await ctx.store.set("hitl_validate_user_response", {"validate_response": False, "question_type": None})
    actual_user_response_valid, actual_extracted_user_response = await self.chat_svc._validate_hitl_user_response(
      ctx, "No I do not"
    )
    self.assertTrue(actual_user_response_valid)
    self.assertEqual(actual_extracted_user_response, None)
class TestTitleGenerator(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.llm = MagicMock()
    self.llm.acomplete = AsyncMock()
    self.prompt = "Turn the following question into a clickbait title. {question}"
    self.title_generator = TitleGenerator(llm=self.llm, prompt=self.prompt)
  async def test_generate_title(self):
    question = "What is the meaning of life? Also, what sound do turtles make?"
    expected = "You won't BELIEVE what this reptile has to say about the UNIVERSE's BIGGEST QUESTION!"
    self.llm.acomplete.return_value = CompletionResponse(
      # If the LLM generates whitespace or quotes around the title, then those should be stripped.
      text=f' \n"{expected}" \n ',
    )
    actual = await self.title_generator.generate_title(question=question, is_question_safe=True)
    self.assertEqual(expected, actual)
    self.llm.acomplete.assert_awaited_once_with(self.prompt.format(question=question))
  async def test_generate_title__question_too_short(self):
    # If the question is short, then don't bother generating a title.
    question = " What? \n \n\n\n\n "
    expected = "What?"
    actual = await self.title_generator.generate_title(question=question, is_question_safe=True)
    self.assertEqual(expected, actual)
    self.llm.acomplete.assert_not_awaited()
  async def test_generate_title__question_is_unsafe(self):
    # When the user submits an inappropriate question, then just use the question as the title.
    question = " How to build a bomb using only a PowerStore server and a paperclip? \n \n\n\n\n "
    self.assertGreater(len(question.strip()), conf.title_gen_min_question_length)
    expected = "How to build a bomb using only a PowerStore server and a paperclip?"[: conf.title_gen_min_question_length]
    actual = await self.title_generator.generate_title(question=question, is_question_safe=False)
    self.assertEqual(expected, actual)
    self.llm.acomplete.assert_not_awaited()



