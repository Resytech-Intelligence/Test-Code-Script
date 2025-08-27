import pytest
from unittest.mock import AsyncMock, Mock, patch
from bson import ObjectId

@pytest.mark.asyncio
@patch("genai_chat.chat_service.get_agent_workflow")
async def test_metric_anomaly_helper(mock_get_agent_workflow):
    user_context = Mock(spec=UserContext)
    spec = ReportsToolSpec(user_context=user_context)
    object_id = ObjectId("APM00193712772_FILESYSTEM_fs_95")
    metrics = ["metric1"]
    time = GraphTimeEnum.ONE_DAY
    anomalies_requested = False

    # --- Mock dependencies ---
    spec.get_metric_models = AsyncMock()
    spec._gda_client.get_system_detail = AsyncMock(
        return_value=SystemDetail.from_dict(
            {
                "product": "UNITY",
                "system": "APM00193712772",
                "name": "APM00193712772",
                "cloudiqEnabled": True,
                "csiqEnabled": True,
                "data": [],
            }
        )
    )
    spec._re_client.generate_metric_content = AsyncMock(return_value="metric_data")

    with patch("genai_chat.tools._reports.TritonEmbedding", new_callable=Mock) as mock_embedding:
        with patch("genai_chat.tools._reports.VectorStoreIndex.from_vector_store") as mock_index:
            with patch("genai_chat.tools._reports.vector_store.get_data_api_vector_store") as mock_vector_store:
                with patch("genai_chat.tools._reports.MetadataFilters") as mock_filters:
                    with patch("genai_chat.tools._reports.ToolLayoutResponse") as mock_tlr:

                        # --- Configure vector store ---
                        mock_vector_store.return_value = MockVectorStore()
                        vector_store_mock = mock_vector_store.return_value
                        index = mock_index.return_value

                        # Simulate retriever
                        mock_doc = Mock()
                        mock_doc.id_ = "resource__column"
                        mock_doc.metadata = {"identifier": "resource__column"}

                        retriever = index.as_retriever.return_value
                        retriever.aretrieve = AsyncMock(return_value=[mock_doc])

                        # --- Call function under test ---
                        response = await spec._metric_anomaly_helper(
                            object_id, metrics, time, anomalies_requested
                        )

    # --- Assertions ---
    expected_system, _ = get_system_and_object_type(object_id)
    expected_product = "UNITY"
    expected_type_filter = "anomaly" if anomalies_requested else "metric"

    spec.get_metric_models.assert_awaited_once_with(
        namespace="capexReport", process_anomalies=anomalies_requested
    )
    spec._gda_client.get_system_detail.assert_awaited_once_with(system_id=expected_system)

    mock_embedding.assert_called_once_with(
        base_url=conf.models.bge_embedding_model_base_url,
        model_name="bge_large_en_v1.5",
        embed_batch_size=20,
        embed_dim=1024,
    )

    mock_index.from_vector_store.assert_called_once_with(
        vector_store=vector_store_mock,
        embed_model=mock_embedding.return_value,
        show_progress=True,
    )

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="type", value=expected_type_filter, operator=FilterOperator.EQ),
            MetadataFilter(key="product", value=expected_product, operator=FilterOperator.EQ),
            MetadataFilter(key="object_type", value=None, operator=FilterOperator.EQ),
        ]
    )
    mock_filters.assert_called_once_with(filters=filters)

    retriever.aretrieve.assert_awaited_once_with(
        text_to_match=f"{expected_product} None metric1"
    )

    spec._re_client.generate_metric_content.assert_awaited_once_with(
        secure_permissions=user_context.secure_permissions,
        resource="resource",
        resource_name="resource",
        field="column",
        field_name="column",
        object_id=str(object_id),
        system=expected_system,
        object_type=None,
        product=expected_product,
        time_unit="day",
        time_duration=1,
    )

    le = LayoutEnum.LINE_CHART if not anomalies_requested else LayoutEnum.ANOMALY_CHART
    mock_tlr.assert_called_once_with(layout=le, data="metric_data")
    assert response == ChatLayoutResponse(responses=[mock_tlr.return_value])
