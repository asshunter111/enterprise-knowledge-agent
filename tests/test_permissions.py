from app.core.permissions import can_access, filter_documents


def test_roles_filter_restricted_documents_before_generation():
    documents = [
        {"metadata": {"department": "public"}},
        {"metadata": {"department": "finance"}},
    ]
    assert len(filter_documents(documents, "employee")) == 1
    assert len(filter_documents(documents, "finance")) == 2
    assert can_access("employee", {"department": "public"})
    assert not can_access("employee", {"department": "finance"})
