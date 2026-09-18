from collections.abc import Iterable

ROLE_ACCESS: dict[str, set[str]] = {
    "employee": {"public"},
    "hr": {"public", "hr"},
    "finance": {"public", "finance"},
    "admin": {"public", "hr", "finance", "admin"},
    "manager": {"public", "manager"},
}


def can_access(role: str, metadata: dict) -> bool:
    access_roles = set(metadata.get("access_roles") or [])
    department = metadata.get("department") or "public"
    if not access_roles and department == "public":
        return True
    allowed = ROLE_ACCESS.get(role, set())
    return department in allowed or bool(access_roles & allowed) or role in access_roles


def filter_documents(documents: Iterable[dict], role: str) -> list[dict]:
    return [document for document in documents if can_access(role, document.get("metadata", {}))]
