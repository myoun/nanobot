from nanobot.cron.targeting import resolve_delivery_target


def test_resolve_delivery_target_for_plain_chat() -> None:
    chat_id, metadata = resolve_delivery_target("telegram", "-1003672808487")

    assert chat_id == "-1003672808487"
    assert metadata == {}


def test_resolve_delivery_target_for_topic_chat() -> None:
    chat_id, metadata = resolve_delivery_target("telegram", "-1003672808487:topic:3")

    assert chat_id == "-1003672808487"
    assert metadata == {"message_thread_id": 3}


def test_resolve_delivery_target_accepts_prefixed_topic_key() -> None:
    chat_id, metadata = resolve_delivery_target("telegram", "telegram:-1003672808487:topic:3")

    assert chat_id == "-1003672808487"
    assert metadata == {"message_thread_id": 3}
