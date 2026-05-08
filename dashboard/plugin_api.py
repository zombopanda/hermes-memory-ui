"""Hermes Memory UI dashboard plugin backend.

Mounted by Hermes dashboard at /api/plugins/hermes-memory-ui/.

Read-only inspection covers built-in memory files, holographic memory,
Mem0, Honcho, and Hindsight provider state. No mutation endpoints are exposed
intentionally. Memory writes should go through Hermes' memory/fact_store
tools or provider classes so validation, locking, FTS/HRR maintenance,
and provider-specific semantics are preserved.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from fastapi import APIRouter, Query
except Exception:  # Allows local syntax/import tests outside the dashboard.
    class APIRouter:  # type: ignore
        def get(self, *_args, **_kwargs):
            return lambda fn: fn

    def Query(default=None, **_kwargs):  # type: ignore
        return default

router = APIRouter()

ENTRY_DELIMITER = "\n§\n"
DEFAULT_MEMORY_LIMIT = 2200
DEFAULT_USER_LIMIT = 1375
DEFAULT_FACT_LIMIT = 500
MAX_FACT_LIMIT = 2000
DEFAULT_MEM0_LIMIT = 500
MAX_MEM0_LIMIT = 2000
DEFAULT_HONCHO_LIMIT = 50
MAX_HONCHO_LIMIT = 100
DEFAULT_HINDSIGHT_LIMIT = 25
MAX_HINDSIGHT_LIMIT = 100
HINDSIGHT_DEFAULT_CLOUD_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_DEFAULT_LOCAL_URL = "http://localhost:8888"
VALID_HINDSIGHT_BUDGETS = {"low", "mid", "high"}


def _hermes_home() -> Path:
    """Return active Hermes home, respecting profiles when available."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()).expanduser()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_simple_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return values


def _env_value(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value not in (None, ""):
        return str(value)
    file_env = _load_simple_env_file(_hermes_home() / ".env")
    return file_env.get(key, default)


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _dig(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _memory_limits(config: Dict[str, Any]) -> Dict[str, int]:
    memory_cfg = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
    # Keep a few aliases to survive config key naming changes across Hermes versions.
    memory_limit = (
        memory_cfg.get("memory_char_limit")
        or memory_cfg.get("memory_limit")
        or memory_cfg.get("agent_memory_char_limit")
        or DEFAULT_MEMORY_LIMIT
    )
    user_limit = (
        memory_cfg.get("user_char_limit")
        or memory_cfg.get("user_profile_char_limit")
        or memory_cfg.get("profile_char_limit")
        or DEFAULT_USER_LIMIT
    )
    try:
        memory_limit = int(memory_limit)
    except Exception:
        memory_limit = DEFAULT_MEMORY_LIMIT
    try:
        user_limit = int(user_limit)
    except Exception:
        user_limit = DEFAULT_USER_LIMIT
    return {"memory": memory_limit, "user": user_limit}


def _parse_entries(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]


def _read_builtin_store(store_id: str, filename: str, label: str, limit: int) -> Dict[str, Any]:
    path = _hermes_home() / "memories" / filename
    try:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        entries = _parse_entries(raw)
        char_count = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        stat = path.stat() if path.exists() else None
        return {
            "id": store_id,
            "label": label,
            "filename": filename,
            "path": str(path),
            "exists": path.exists(),
            "entries": entries,
            "entry_count": len(entries),
            "char_count": char_count,
            "char_limit": limit,
            "usage_percent": round((char_count / limit) * 100, 1) if limit else None,
            "modified_at": stat.st_mtime if stat else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "id": store_id,
            "label": label,
            "filename": filename,
            "path": str(path),
            "exists": path.exists(),
            "entries": [],
            "entry_count": 0,
            "char_count": 0,
            "char_limit": limit,
            "usage_percent": 0,
            "modified_at": None,
            "error": str(exc),
        }


def _builtin_payload(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    limits = _memory_limits(config)
    stores = [
        _read_builtin_store("memory", "MEMORY.md", "Agent memory", limits["memory"]),
        _read_builtin_store("user", "USER.md", "User profile", limits["user"]),
    ]
    return {
        "hermes_home": str(_hermes_home()),
        "stores": stores,
        "total_entries": sum(s["entry_count"] for s in stores),
        "generated_at": time.time(),
    }


def _resolve_holographic_db(config: Dict[str, Any]) -> Path:
    home = _hermes_home()
    db_path = _dig(config, "plugins", "hermes-memory-store", "db_path", default=None)
    if not db_path:
        db_path = str(home / "memory_store.db")
    if isinstance(db_path, str):
        db_path = db_path.replace("$HERMES_HOME", str(home)).replace("${HERMES_HOME}", str(home))
        return Path(db_path).expanduser()
    return home / "memory_store.db"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    # SQLite URI read-only mode prevents accidental writes from this dashboard plugin.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_like(text: str) -> str:
    return f"%{text.replace('%', '').replace('_', '')}%"


def _holographic_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    limit: int = DEFAULT_FACT_LIMIT,
    category: Optional[str] = None,
    min_trust: float = 0.0,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    db_path = _resolve_holographic_db(config)
    provider = _dig(config, "memory", "provider", default=None)
    category = category if isinstance(category, str) and category.strip() else None
    search = search if isinstance(search, str) and search.strip() else None
    try:
        min_trust = float(min_trust or 0.0)
    except Exception:
        min_trust = 0.0
    try:
        limit = int(limit or DEFAULT_FACT_LIMIT)
    except Exception:
        limit = DEFAULT_FACT_LIMIT
    limit = max(1, min(limit, MAX_FACT_LIMIT))

    base: Dict[str, Any] = {
        "id": "holographic",
        "label": "Holographic memory",
        "provider_configured": provider == "holographic",
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "facts": [],
        "fact_count": 0,
        "total_facts": 0,
        "categories": [],
        "entities_count": 0,
        "memory_banks_count": 0,
        "limit": limit,
        "error": None,
        "generated_at": time.time(),
    }

    if not db_path.exists():
        return base

    try:
        with _connect_readonly(db_path) as conn:
            try:
                base["total_facts"] = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            except Exception:
                base["total_facts"] = 0
            try:
                base["entities_count"] = int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
            except Exception:
                base["entities_count"] = 0
            try:
                base["memory_banks_count"] = int(conn.execute("SELECT COUNT(*) FROM memory_banks").fetchone()[0])
            except Exception:
                base["memory_banks_count"] = 0
            try:
                rows = conn.execute(
                    """
                    SELECT category, COUNT(*) AS count
                    FROM facts
                    GROUP BY category
                    ORDER BY count DESC, category ASC
                    """
                ).fetchall()
                base["categories"] = [{"category": r["category"] or "general", "count": r["count"]} for r in rows]
            except Exception:
                base["categories"] = []

            where = ["trust_score >= ?"]
            params: List[Any] = [float(min_trust or 0.0)]
            if category:
                where.append("category = ?")
                params.append(category)
            if search:
                where.append("(content LIKE ? OR tags LIKE ?)")
                like = _safe_like(search)
                params.extend([like, like])
            params.append(limit)
            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE {' AND '.join(where)}
                ORDER BY fact_id ASC
                LIMIT ?
            """
            facts = [dict(row) for row in conn.execute(sql, params).fetchall()]
            base["facts"] = facts
            base["fact_count"] = len(facts)
    except Exception as exc:
        base["error"] = str(exc)

    return base


def _expand_path(value: Optional[str], home: Optional[Path] = None) -> Optional[Path]:
    if not value or not isinstance(value, str):
        return None
    home = home or _hermes_home()
    expanded = value.replace("$HERMES_HOME", str(home)).replace("${HERMES_HOME}", str(home))
    return Path(expanded).expanduser()


def _load_mem0_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load non-secret Mem0 configuration for read-only dashboard access."""
    home = _hermes_home()
    config_path = home / "mem0.json"
    file_cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8")) or {}
            file_cfg = data if isinstance(data, dict) else {}
        except Exception:
            file_cfg = {}

    def pick(key: str, env_key: str, default: Any = None) -> Any:
        value = os.environ.get(env_key, default)
        if key in file_cfg and file_cfg.get(key) not in (None, ""):
            value = file_cfg.get(key)
        return value

    api_key = pick("api_key", "MEM0_API_KEY", "")
    rerank = pick("rerank", "MEM0_RERANK", True)
    if isinstance(rerank, str):
        rerank = rerank.strip().lower() not in {"0", "false", "no", "off"}
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "api_key_present": bool(api_key),
        "user_id": pick("user_id", "MEM0_USER_ID", "hermes-user"),
        "agent_id": pick("agent_id", "MEM0_AGENT_ID", "hermes"),
        "rerank": rerank,
        # Keep the real value private and local to the API call path.
        "_api_key": api_key,
    }


def _unwrap_mem0_results(response: Any) -> List[Any]:
    if isinstance(response, dict):
        results = response.get("results", response.get("memories", []))
        return results if isinstance(results, list) else []
    if isinstance(response, list):
        return response
    return []


def _normalize_mem0_memory(item: Any, index: int) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"id": str(index + 1), "memory": item, "score": None, "created_at": None, "updated_at": None, "metadata": {}}
    if not isinstance(item, dict):
        return {"id": str(index + 1), "memory": str(item), "score": None, "created_at": None, "updated_at": None, "metadata": {}}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "id": item.get("id") or item.get("memory_id") or item.get("uuid") or str(index + 1),
        "memory": item.get("memory") or item.get("text") or item.get("content") or "",
        "score": item.get("score"),
        "created_at": item.get("created_at") or item.get("createdAt"),
        "updated_at": item.get("updated_at") or item.get("updatedAt"),
        "user_id": item.get("user_id") or item.get("userId"),
        "agent_id": item.get("agent_id") or item.get("agentId"),
        "metadata": metadata,
    }


def _filter_mem0_memories(memories: List[Dict[str, Any]], search: Optional[str], limit: int) -> List[Dict[str, Any]]:
    if search:
        needle = search.casefold()
        memories = [m for m in memories if needle in str(m.get("memory", "")).casefold()]
    return memories[:limit]


def _mem0_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    limit: int = DEFAULT_MEM0_LIMIT,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    provider = _dig(config, "memory", "provider", default=None)
    mem0_cfg = _load_mem0_config(config)
    search = search if isinstance(search, str) and search.strip() else None
    try:
        limit = int(limit or DEFAULT_MEM0_LIMIT)
    except Exception:
        limit = DEFAULT_MEM0_LIMIT
    limit = max(1, min(limit, MAX_MEM0_LIMIT))

    base: Dict[str, Any] = {
        "id": "mem0",
        "label": "Mem0 memory",
        "provider_configured": provider == "mem0",
        "mode": "read-only",
        "config_path": mem0_cfg["config_path"],
        "config_exists": mem0_cfg["config_exists"],
        "api_key_present": mem0_cfg["api_key_present"],
        "user_id": mem0_cfg["user_id"],
        "agent_id": mem0_cfg["agent_id"],
        "memories": [],
        "memory_count": 0,
        "total_memories": 0,
        "limit": limit,
        "search": search or "",
        "error": None,
        "generated_at": time.time(),
    }

    try:
        if not mem0_cfg["api_key_present"]:
            base["error"] = "Mem0 API key not configured. Set MEM0_API_KEY in $HERMES_HOME/.env or the process environment."
            return base

        try:
            from mem0 import MemoryClient  # type: ignore
        except ImportError:
            base["error"] = "mem0 package not installed in the dashboard environment. Install mem0ai."
            return base

        client = MemoryClient(api_key=mem0_cfg["_api_key"])
        filters = {"user_id": mem0_cfg["user_id"]}
        if search:
            response = client.search(query=search, filters=filters, rerank=mem0_cfg["rerank"], top_k=limit)
        else:
            response = client.get_all(filters=filters)
        all_memories = [_normalize_mem0_memory(item, index) for index, item in enumerate(_unwrap_mem0_results(response))]
        base["total_memories"] = len(all_memories)
        base["memories"] = _filter_mem0_memories(all_memories, None, limit)
        base["memory_count"] = len(base["memories"])
    except Exception as exc:
        base["error"] = str(exc)

    return base


def _normalize_honcho_card(card: Any) -> List[str]:
    if not card:
        return []
    if isinstance(card, (list, tuple)):
        return [str(item) for item in card if item]
    return [str(card)]


def _object_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        try:
            data = model_dump()
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    as_dict = getattr(item, "dict", None)
    if callable(as_dict):
        try:
            data = as_dict()
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    result: Dict[str, Any] = {}
    for key in ("id", "content", "created_at", "updated_at", "session_id", "metadata"):
        if hasattr(item, key):
            value = getattr(item, key)
            if not callable(value):
                result[key] = value
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    return str(value)


def _normalize_honcho_conclusion(item: Any, index: int) -> Dict[str, Any]:
    data = _object_to_dict(item)
    content = data.get("content") or data.get("text") or data.get("body") or ""
    return {
        "id": data.get("id") or data.get("conclusion_id") or data.get("uuid") or str(index + 1),
        "content": str(content),
        "created_at": _json_safe(data.get("created_at") or data.get("createdAt")),
        "updated_at": _json_safe(data.get("updated_at") or data.get("updatedAt")),
        "session_id": data.get("session_id") or data.get("sessionId"),
        "metadata": _json_safe(data.get("metadata") if isinstance(data.get("metadata"), dict) else {}),
    }


def _honcho_config_payload(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    provider = _dig(config, "memory", "provider", default=None)
    fallback_path = _hermes_home() / "honcho.json"
    base: Dict[str, Any] = {
        "provider_configured": provider == "honcho",
        "config_path": str(fallback_path),
        "config_exists": fallback_path.exists(),
        "api_key_present": bool(os.environ.get("HONCHO_API_KEY")),
        "base_url_present": bool(os.environ.get("HONCHO_BASE_URL")),
        "enabled": False,
        "host": os.environ.get("HERMES_HONCHO_HOST", "hermes"),
        "workspace": "hermes",
        "user_peer": "user",
        "ai_peer": "hermes",
        "environment": os.environ.get("HONCHO_ENVIRONMENT", "production"),
        "recall_mode": "hybrid",
        "session_strategy": "per-directory",
        "save_messages": None,
        "write_frequency": None,
        "context_tokens": None,
        "dialectic_depth": None,
        "dialectic_reasoning_level": None,
        "dialectic_dynamic": None,
        "dialectic_max_chars": None,
        "observation_mode": None,
        "user_observe_me": None,
        "user_observe_others": None,
        "ai_observe_me": None,
        "ai_observe_others": None,
        "explicitly_configured": False,
        "_client_config": None,
        "_import_error": None,
    }
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path  # type: ignore

        cfg = HonchoClientConfig.from_global_config()
        path = resolve_config_path()
        base.update({
            "config_path": str(path),
            "config_exists": Path(path).exists(),
            "api_key_present": bool(getattr(cfg, "api_key", None)),
            "base_url_present": bool(getattr(cfg, "base_url", None)),
            "enabled": bool(getattr(cfg, "enabled", False)),
            "host": getattr(cfg, "host", None) or "hermes",
            "workspace": getattr(cfg, "workspace_id", None) or "hermes",
            "user_peer": getattr(cfg, "peer_name", None) or "user",
            "ai_peer": getattr(cfg, "ai_peer", None) or getattr(cfg, "host", None) or "hermes",
            "environment": getattr(cfg, "environment", None) or "production",
            "recall_mode": getattr(cfg, "recall_mode", None) or "hybrid",
            "session_strategy": getattr(cfg, "session_strategy", None) or "per-directory",
            "save_messages": getattr(cfg, "save_messages", None),
            "write_frequency": getattr(cfg, "write_frequency", None),
            "context_tokens": getattr(cfg, "context_tokens", None),
            "dialectic_depth": getattr(cfg, "dialectic_depth", None),
            "dialectic_reasoning_level": getattr(cfg, "dialectic_reasoning_level", None),
            "dialectic_dynamic": getattr(cfg, "dialectic_dynamic", None),
            "dialectic_max_chars": getattr(cfg, "dialectic_max_chars", None),
            "observation_mode": getattr(cfg, "observation_mode", None),
            "user_observe_me": getattr(cfg, "user_observe_me", None),
            "user_observe_others": getattr(cfg, "user_observe_others", None),
            "ai_observe_me": getattr(cfg, "ai_observe_me", None),
            "ai_observe_others": getattr(cfg, "ai_observe_others", None),
            "explicitly_configured": getattr(cfg, "explicitly_configured", False),
            "_client_config": cfg,
        })
    except Exception as exc:
        base["_import_error"] = str(exc)
    return base


def _call_peer_context(peer_obj: Any, *, target: str, search: Optional[str], limit: int) -> Dict[str, Any]:
    representation = ""
    card: List[str] = []
    try:
        kwargs: Dict[str, Any] = {"target": target}
        if search:
            kwargs["search_query"] = search
            kwargs["search_top_k"] = limit
        try:
            ctx = peer_obj.context(**kwargs)
        except TypeError:
            kwargs.pop("search_top_k", None)
            ctx = peer_obj.context(**kwargs)
        representation = getattr(ctx, "representation", None) or getattr(ctx, "peer_representation", None) or ""
        card = _normalize_honcho_card(getattr(ctx, "peer_card", None))
    except Exception:
        pass
    if not representation:
        try:
            representation = peer_obj.representation(target=target) or ""
        except Exception:
            representation = ""
    if not card:
        try:
            getter = getattr(peer_obj, "get_card", None) or getattr(peer_obj, "card", None)
            if callable(getter):
                card = _normalize_honcho_card(getter(target=target))
        except Exception:
            card = []
    return {"representation": str(representation or ""), "card": card}


def _list_honcho_conclusions(observer_peer: Any, target_peer_id: str, limit: int) -> List[Dict[str, Any]]:
    try:
        scope = observer_peer.conclusions_of(target_peer_id)
        try:
            items = scope.list(page=1, size=limit, reverse=True)
        except TypeError:
            items = scope.list(page=1, size=limit)
        if not isinstance(items, list):
            items = list(items or [])
        return [_normalize_honcho_conclusion(item, index) for index, item in enumerate(items[:limit])]
    except Exception:
        return []


def _honcho_search_results(base: Dict[str, Any], search: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Return visible, deterministic text matches for the dashboard search box.

    Honcho's `peer.context(search_query=...)` may still return the same peer card
    shape, especially with local/self-hosted demo data or missing embeddings. The
    dashboard should nevertheless show that Apply/Refresh used the submitted
    query, so expose lightweight read-only matches from the already returned card
    and conclusion text.
    """
    if not search:
        return []
    needle = search.casefold()
    results: List[Dict[str, Any]] = []
    for scope_key, label in (("user", "User peer"), ("ai", "AI peer")):
        peer = base.get(scope_key, {}) if isinstance(base.get(scope_key), dict) else {}
        peer_id = peer.get("peer_id") or scope_key
        for index, item in enumerate(peer.get("card") or []):
            text = str(item)
            if needle in text.casefold():
                results.append({"source": f"{label} card", "peer_id": peer_id, "id": str(index + 1), "content": text})
        representation = str(peer.get("representation") or "")
        if representation and needle in representation.casefold():
            results.append({"source": f"{label} representation", "peer_id": peer_id, "id": "representation", "content": representation})
        for conclusion in peer.get("conclusions") or []:
            text = str(conclusion.get("content", "")) if isinstance(conclusion, dict) else str(conclusion)
            if needle in text.casefold():
                results.append({
                    "source": f"{label} conclusion",
                    "peer_id": peer_id,
                    "id": str(conclusion.get("id", len(results) + 1)) if isinstance(conclusion, dict) else str(len(results) + 1),
                    "content": text,
                })
    return results[:limit]


def _honcho_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    limit: int = DEFAULT_HONCHO_LIMIT,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    honcho_cfg = _honcho_config_payload(config)
    search = search if isinstance(search, str) and search.strip() else None
    try:
        limit = int(limit or DEFAULT_HONCHO_LIMIT)
    except Exception:
        limit = DEFAULT_HONCHO_LIMIT
    limit = max(1, min(limit, MAX_HONCHO_LIMIT))
    base: Dict[str, Any] = {
        "id": "honcho",
        "label": "Honcho memory",
        "provider_configured": honcho_cfg["provider_configured"],
        "mode": "read-only",
        "config_path": honcho_cfg["config_path"],
        "config_exists": honcho_cfg["config_exists"],
        "api_key_present": honcho_cfg["api_key_present"],
        "base_url_present": honcho_cfg["base_url_present"],
        "enabled": honcho_cfg["enabled"],
        "host": honcho_cfg["host"],
        "workspace": honcho_cfg["workspace"],
        "user_peer": honcho_cfg["user_peer"],
        "ai_peer": honcho_cfg["ai_peer"],
        "environment": honcho_cfg["environment"],
        "recall_mode": honcho_cfg["recall_mode"],
        "session_strategy": honcho_cfg["session_strategy"],
        "save_messages": honcho_cfg["save_messages"],
        "write_frequency": honcho_cfg["write_frequency"],
        "context_tokens": honcho_cfg["context_tokens"],
        "dialectic_depth": honcho_cfg["dialectic_depth"],
        "dialectic_reasoning_level": honcho_cfg["dialectic_reasoning_level"],
        "dialectic_dynamic": honcho_cfg["dialectic_dynamic"],
        "dialectic_max_chars": honcho_cfg["dialectic_max_chars"],
        "observation_mode": honcho_cfg["observation_mode"],
        "user_observe_me": honcho_cfg["user_observe_me"],
        "user_observe_others": honcho_cfg["user_observe_others"],
        "ai_observe_me": honcho_cfg["ai_observe_me"],
        "ai_observe_others": honcho_cfg["ai_observe_others"],
        "explicitly_configured": honcho_cfg["explicitly_configured"],
        "user": {"peer_id": honcho_cfg["user_peer"], "card": [], "representation": "", "conclusions": []},
        "ai": {"peer_id": honcho_cfg["ai_peer"], "card": [], "representation": "", "conclusions": []},
        "search_results": [],
        "search_result_count": 0,
        "limit": limit,
        "search": search or "",
        "error": None,
        "generated_at": time.time(),
    }
    cfg = honcho_cfg.get("_client_config")
    if cfg is None:
        base["error"] = honcho_cfg.get("_import_error") or "Honcho provider helpers are not available in the dashboard environment."
        return base
    if not (honcho_cfg["api_key_present"] or honcho_cfg["base_url_present"]):
        base["error"] = "Honcho API key or base URL is not configured. Run 'hermes honcho setup' or set HONCHO_API_KEY / HONCHO_BASE_URL."
        return base
    try:
        from plugins.memory.honcho.client import get_honcho_client  # type: ignore

        client = get_honcho_client(cfg)
        user_peer_id = str(honcho_cfg["user_peer"] or "user")
        ai_peer_id = str(honcho_cfg["ai_peer"] or honcho_cfg["host"] or "hermes")
        user_peer_obj = client.peer(user_peer_id)
        ai_peer_obj = client.peer(ai_peer_id)
        base["user"].update(_call_peer_context(user_peer_obj, target=user_peer_id, search=search, limit=limit))
        base["ai"].update(_call_peer_context(ai_peer_obj, target=ai_peer_id, search=search, limit=limit))
        base["user"]["conclusions"] = _list_honcho_conclusions(ai_peer_obj, user_peer_id, limit)
        base["ai"]["conclusions"] = _list_honcho_conclusions(ai_peer_obj, ai_peer_id, limit)
        base["search_results"] = _honcho_search_results(base, search, limit)
        base["search_result_count"] = len(base["search_results"])
    except Exception as exc:
        base["error"] = str(exc)
    return base


def _load_hindsight_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load non-secret Hindsight configuration for dashboard inspection."""
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    home = _hermes_home()
    profile_path = home / "hindsight" / "config.json"
    legacy_path = Path.home() / ".hindsight" / "config.json"
    file_cfg: Dict[str, Any] = {}
    config_path = profile_path
    config_exists = profile_path.exists()
    if config_exists:
        file_cfg = _read_json(profile_path)
    elif legacy_path.exists():
        config_path = legacy_path
        config_exists = True
        file_cfg = _read_json(legacy_path)

    mode = str(file_cfg.get("mode") or _env_value("HINDSIGHT_MODE", "cloud") or "cloud")
    if mode == "local":
        mode = "local_embedded"
    api_key = file_cfg.get("apiKey") or file_cfg.get("api_key") or _env_value("HINDSIGHT_API_KEY", "")
    llm_key = file_cfg.get("llmApiKey") or file_cfg.get("llm_api_key") or _env_value("HINDSIGHT_LLM_API_KEY", "")
    default_url = HINDSIGHT_DEFAULT_LOCAL_URL if mode in {"local_embedded", "local_external"} else HINDSIGHT_DEFAULT_CLOUD_URL
    api_url = file_cfg.get("api_url") or _env_value("HINDSIGHT_API_URL", default_url)
    banks = file_cfg.get("banks") if isinstance(file_cfg.get("banks"), dict) else {}
    hermes_bank = banks.get("hermes") if isinstance(banks.get("hermes"), dict) else {}
    bank_id = file_cfg.get("bank_id") or hermes_bank.get("bankId") or _env_value("HINDSIGHT_BANK_ID", "hermes")
    bank_template = file_cfg.get("bank_id_template", "") or ""
    budget = file_cfg.get("recall_budget") or file_cfg.get("budget") or hermes_bank.get("budget") or _env_value("HINDSIGHT_BUDGET", "mid")
    if budget not in VALID_HINDSIGHT_BUDGETS:
        budget = "mid"
    return {
        "provider_configured": _dig(config, "memory", "provider", default=None) == "hindsight",
        "config_path": str(config_path),
        "config_exists": bool(config_exists),
        "mode": mode,
        "api_url": str(api_url),
        "api_key_present": bool(api_key),
        "llm_key_present": bool(llm_key),
        "llm_provider": file_cfg.get("llm_provider") or "",
        "llm_model": file_cfg.get("llm_model") or "",
        "llm_base_url_present": bool(file_cfg.get("llm_base_url") or _env_value("HINDSIGHT_API_LLM_BASE_URL", "")),
        "bank_id": str(bank_id or "hermes"),
        "bank_id_template": str(bank_template),
        "bank_mission": file_cfg.get("bank_mission", ""),
        "bank_retain_mission": file_cfg.get("bank_retain_mission") or "",
        "recall_budget": budget,
        "recall_prefetch_method": file_cfg.get("recall_prefetch_method") or file_cfg.get("prefetch_method") or "recall",
        "recall_max_tokens": file_cfg.get("recall_max_tokens", 4096),
        "recall_max_input_chars": file_cfg.get("recall_max_input_chars", 800),
        "recall_tags": file_cfg.get("recall_tags") or None,
        "recall_tags_match": file_cfg.get("recall_tags_match", "any"),
        "memory_mode": file_cfg.get("memory_mode", "hybrid"),
        "auto_retain": _truthy(file_cfg.get("auto_retain"), True),
        "auto_recall": _truthy(file_cfg.get("auto_recall"), True),
        "retain_async": _truthy(file_cfg.get("retain_async"), True),
        "retain_every_n_turns": file_cfg.get("retain_every_n_turns", 1),
        "timeout": file_cfg.get("timeout") if file_cfg.get("timeout") is not None else _env_value("HINDSIGHT_TIMEOUT", "120"),
        "idle_timeout": file_cfg.get("idle_timeout") if file_cfg.get("idle_timeout") is not None else _env_value("HINDSIGHT_IDLE_TIMEOUT", "300"),
        "profile": file_cfg.get("profile", "hermes"),
        "_api_key": api_key,
        "_file_config": file_cfg,
    }


def _normalize_hindsight_result(item: Any, index: int) -> Dict[str, Any]:
    data = _object_to_dict(item)

    def attr(name: str, default: Any = None) -> Any:
        if name in data:
            return data.get(name)
        value = getattr(item, name, default)
        return default if callable(value) else value

    text = attr("text") or attr("content") or attr("memory") or attr("document") or ""
    metadata = attr("metadata", {})
    return {
        "id": attr("id") or attr("document_id") or attr("uuid") or str(index + 1),
        "text": str(text),
        "score": attr("score") or attr("relevance") or attr("similarity"),
        "type": attr("type") or attr("kind"),
        "metadata": _json_safe(metadata if isinstance(metadata, dict) else {}),
    }


def _hindsight_should_manage_local_daemon(cfg: Dict[str, Any]) -> bool:
    if cfg.get("mode") != "local_embedded":
        return False
    parsed = urllib.parse.urlparse(str(cfg.get("api_url") or HINDSIGHT_DEFAULT_LOCAL_URL))
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _ensure_hindsight_local_daemon(cfg: Dict[str, Any]) -> Optional[str]:
    """Best-effort start for local_embedded Hindsight before client calls."""
    if not _hindsight_should_manage_local_daemon(cfg):
        return None
    profile = str(cfg.get("profile") or "hermes")
    cmd = ["hindsight-embed", "-p", profile, "daemon", "start"]
    env = os.environ.copy()
    env.setdefault("HERMES_HOME", str(_hermes_home()))
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return "hindsight-embed command not found in dashboard environment"
    except Exception as exc:
        return f"Could not start local Hindsight daemon: {exc}"
    if result.returncode not in (0,):
        output = (result.stderr or result.stdout or "").strip()
        return output or f"hindsight-embed daemon start exited with {result.returncode}"
    return None


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine from sync dashboard helper code, even inside FastAPI's event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _hindsight_timeout_seconds(cfg: Dict[str, Any]) -> float:
    try:
        return float(cfg.get("timeout") or 120)
    except Exception:
        return 120.0


def _hindsight_connection_failed(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "connection refused" in text or "cannot connect" in text or "connect call failed" in text


def _hindsight_client_call(cfg: Dict[str, Any], fn: Callable[[Any], Any]) -> Any:
    """Call the official Hindsight client for read-only inspection operations."""

    async def invoke() -> Any:
        from hindsight_client import Hindsight  # type: ignore

        client = Hindsight(
            base_url=str(cfg.get("api_url") or HINDSIGHT_DEFAULT_LOCAL_URL).rstrip("/"),
            api_key=cfg.get("_api_key") or None,
            timeout=_hindsight_timeout_seconds(cfg),
            user_agent="hermes-memory-ui-dashboard/0.4.6",
        )
        try:
            return await fn(client)
        finally:
            await client.aclose()

    try:
        return _run_coro_blocking(invoke())
    except Exception as exc:
        if _hindsight_connection_failed(exc):
            start_error = _ensure_hindsight_local_daemon(cfg)
            if start_error:
                raise RuntimeError(start_error) from exc
            return _run_coro_blocking(invoke())
        raise


def _normalize_hindsight_document(item: Any, index: int) -> Dict[str, Any]:
    data = item if isinstance(item, dict) else _object_to_dict(item)

    def attr(name: str, default: Any = None) -> Any:
        if isinstance(data, dict) and name in data:
            return data.get(name)
        value = getattr(item, name, default)
        return default if callable(value) else value

    text = attr("original_text") or attr("text") or attr("content") or ""
    doc_metadata = attr("document_metadata", {}) or attr("metadata", {}) or {}
    return {
        "id": attr("id") or attr("document_id") or str(index + 1),
        "text": str(text),
        "type": "document",
        "score": None,
        "metadata": {
            "source": "hindsight_client_documents",
            "memory_unit_count": attr("memory_unit_count"),
            "text_length": attr("text_length") or len(str(text)),
            "created_at": _json_safe(attr("created_at")),
            "updated_at": _json_safe(attr("updated_at")),
            "tags": attr("tags", []) or [],
            "retain_params": _json_safe(attr("retain_params")),
            "document_metadata": _json_safe(doc_metadata if isinstance(doc_metadata, dict) else {}),
        },
    }


def _hindsight_contents_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    limit: int = DEFAULT_HINDSIGHT_LIMIT,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List visible Hindsight memory units and source documents via hindsight_client."""
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    cfg = _load_hindsight_config(config)
    try:
        limit = int(limit or DEFAULT_HINDSIGHT_LIMIT)
    except Exception:
        limit = DEFAULT_HINDSIGHT_LIMIT
    limit = max(1, min(limit, MAX_HINDSIGHT_LIMIT))
    search = search if isinstance(search, str) and search.strip() else None
    bank_id = str(cfg.get("bank_id") or "hermes")
    base = _hindsight_config_payload(config)
    base.update({
        "operation": "contents",
        "search": search or "",
        "limit": limit,
        "memories": [],
        "memory_count": 0,
        "total_memories": 0,
        "documents": [],
        "document_count": 0,
        "total_documents": 0,
        "stats": {},
    })
    try:
        async def fetch_contents(client: Any) -> Dict[str, Any]:
            stats_resp = await client.banks.get_agent_stats(
                bank_id=bank_id,
                _request_timeout=_hindsight_timeout_seconds(cfg),
            )
            memories_resp = await client.memory.list_memories(
                bank_id=bank_id,
                q=search,
                limit=limit,
                offset=0,
                _request_timeout=_hindsight_timeout_seconds(cfg),
            )
            docs_resp = await client.documents.list_documents(
                bank_id=bank_id,
                q=None,
                limit=max(limit, 100),
                offset=0,
                _request_timeout=_hindsight_timeout_seconds(cfg),
            )
            docs_items = list(getattr(docs_resp, "items", None) or [])
            detailed_docs: List[Dict[str, Any]] = []
            for item in docs_items[: max(limit, 100)]:
                doc_id = getattr(item, "id", None) or (_object_to_dict(item).get("id") if item is not None else None)
                doc_data = item
                if doc_id:
                    try:
                        doc_data = await client.documents.get_document(
                            bank_id=bank_id,
                            document_id=str(doc_id),
                            _request_timeout=_hindsight_timeout_seconds(cfg),
                        )
                    except Exception:
                        doc_data = item
                normalized = _normalize_hindsight_document(doc_data, len(detailed_docs))
                if search:
                    haystack = (normalized.get("text", "") + " " + normalized.get("id", "") + " " + json.dumps(normalized.get("metadata", {}))).casefold()
                    if search.casefold() not in haystack:
                        continue
                detailed_docs.append(normalized)
                if len(detailed_docs) >= limit:
                    break
            memories_items = list(getattr(memories_resp, "items", None) or [])
            return {
                "stats": _json_safe(_object_to_dict(stats_resp)),
                "memories": [_normalize_hindsight_result(item, index) for index, item in enumerate(memories_items[:limit])],
                "total_memories": getattr(memories_resp, "total", len(memories_items)) or len(memories_items),
                "documents": detailed_docs,
                "total_documents": getattr(docs_resp, "total", len(docs_items)) or len(docs_items),
            }

        content = _hindsight_client_call(cfg, fetch_contents)
        base["stats"] = content.get("stats", {})
        base["memories"] = content.get("memories", [])
        base["memory_count"] = len(base["memories"])
        base["total_memories"] = content.get("total_memories", base["memory_count"])
        base["documents"] = content.get("documents", [])
        base["document_count"] = len(base["documents"])
        base["total_documents"] = content.get("total_documents", base["document_count"])
    except Exception as exc:
        base["error"] = str(exc)
    return base


def _make_hindsight_provider() -> Any:
    from plugins.memory.hindsight import HindsightMemoryProvider  # type: ignore

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="dashboard", hermes_home=str(_hermes_home()), platform="dashboard")
    return provider


def _hindsight_config_payload(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _load_hindsight_config(config)
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    public.update({
        "id": "hindsight",
        "label": "Hindsight memory",
        "mode_label": "query-only",
        "generated_at": time.time(),
        "error": None,
    })
    return public


def _hindsight_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    query: Optional[str] = None,
    limit: int = DEFAULT_HINDSIGHT_LIMIT,
    mode: str = "status",
) -> Dict[str, Any]:
    config = config if config is not None else _read_yaml(_hermes_home() / "config.yaml")
    query = query if isinstance(query, str) and query.strip() else None
    try:
        limit = int(limit or DEFAULT_HINDSIGHT_LIMIT)
    except Exception:
        limit = DEFAULT_HINDSIGHT_LIMIT
    limit = max(1, min(limit, MAX_HINDSIGHT_LIMIT))
    base = _hindsight_config_payload(config)
    base.update({
        "operation": mode,
        "query": query or "",
        "limit": limit,
        "results": [],
        "result_count": 0,
        "reflection": "",
    })
    if mode == "status":
        return base
    if mode not in {"recall", "reflect"}:
        base["error"] = f"Unsupported Hindsight operation: {mode}"
        return base
    if not query:
        base["error"] = "Query is required for Hindsight recall/reflect."
        return base
    provider = None
    try:
        provider = _make_hindsight_provider()
        if mode == "reflect":
            response = provider._run_hindsight_operation(
                lambda client: client.areflect(bank_id=provider._bank_id, query=query, budget=provider._budget)
            )
            base["reflection_source"] = "hindsight_reflect"
            base["reflection"] = str(getattr(response, "text", "") or "")
            return base
        recall_kwargs: Dict[str, Any] = {
            "bank_id": provider._bank_id,
            "query": query,
            "budget": provider._budget,
            "max_tokens": provider._recall_max_tokens,
        }
        if getattr(provider, "_recall_tags", None):
            recall_kwargs["tags"] = provider._recall_tags
            recall_kwargs["tags_match"] = provider._recall_tags_match
        if getattr(provider, "_recall_types", None):
            recall_kwargs["types"] = provider._recall_types
        response = provider._run_hindsight_operation(lambda client: client.arecall(**recall_kwargs))
        raw_results = list(getattr(response, "results", None) or [])[:limit]
        base["results"] = [_normalize_hindsight_result(item, index) for index, item in enumerate(raw_results)]
        base["result_source"] = "hindsight_recall"
        base["result_count"] = len(base["results"])
    except Exception as exc:
        base["error"] = str(exc)
    finally:
        shutdown = getattr(provider, "shutdown", None) if provider is not None else None
        if callable(shutdown):
            try:
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    shutdown()
            except Exception:
                pass
    return base


@router.get("/status")
async def status() -> Dict[str, Any]:
    home = _hermes_home()
    config = _read_yaml(home / "config.yaml")
    db_path = _resolve_holographic_db(config)
    mem0_cfg = _load_mem0_config(config)
    honcho_cfg = _honcho_config_payload(config)
    hindsight_cfg = _load_hindsight_config(config)
    return {
        "plugin": "hermes-memory-ui",
        "version": "0.4.6",
        "mode": "read-only",
        "hermes_home": str(home),
        "config_path": str(home / "config.yaml"),
        "memory_provider": _dig(config, "memory", "provider", default=None),
        "builtin": {
            "memory_path": str(home / "memories" / "MEMORY.md"),
            "memory_exists": (home / "memories" / "MEMORY.md").exists(),
            "user_path": str(home / "memories" / "USER.md"),
            "user_exists": (home / "memories" / "USER.md").exists(),
        },
        "holographic": {
            "db_path": str(db_path),
            "db_exists": db_path.exists(),
            "provider_configured": _dig(config, "memory", "provider", default=None) == "holographic",
        },
        "mem0": {
            "config_path": mem0_cfg["config_path"],
            "config_exists": mem0_cfg["config_exists"],
            "api_key_present": mem0_cfg["api_key_present"],
            "user_id": mem0_cfg["user_id"],
            "agent_id": mem0_cfg["agent_id"],
            "provider_configured": _dig(config, "memory", "provider", default=None) == "mem0",
        },
        "honcho": {
            "config_path": honcho_cfg["config_path"],
            "config_exists": honcho_cfg["config_exists"],
            "api_key_present": honcho_cfg["api_key_present"],
            "base_url_present": honcho_cfg["base_url_present"],
            "enabled": honcho_cfg["enabled"],
            "host": honcho_cfg["host"],
            "workspace": honcho_cfg["workspace"],
            "user_peer": honcho_cfg["user_peer"],
            "ai_peer": honcho_cfg["ai_peer"],
            "recall_mode": honcho_cfg["recall_mode"],
            "session_strategy": honcho_cfg["session_strategy"],
            "provider_configured": _dig(config, "memory", "provider", default=None) == "honcho",
        },
        "hindsight": {
            "config_path": hindsight_cfg["config_path"],
            "config_exists": hindsight_cfg["config_exists"],
            "mode": hindsight_cfg["mode"],
            "api_url": hindsight_cfg["api_url"],
            "api_key_present": hindsight_cfg["api_key_present"],
            "llm_key_present": hindsight_cfg["llm_key_present"],
            "llm_provider": hindsight_cfg["llm_provider"],
            "llm_model": hindsight_cfg["llm_model"],
            "bank_id": hindsight_cfg["bank_id"],
            "bank_id_template": hindsight_cfg["bank_id_template"],
            "recall_budget": hindsight_cfg["recall_budget"],
            "memory_mode": hindsight_cfg["memory_mode"],
            "auto_retain": hindsight_cfg["auto_retain"],
            "auto_recall": hindsight_cfg["auto_recall"],
            "provider_configured": hindsight_cfg["provider_configured"],
        },
        "generated_at": time.time(),
    }


@router.get("/builtin")
async def builtin() -> Dict[str, Any]:
    return _builtin_payload()


@router.get("/holographic")
async def holographic(
    limit: int = Query(DEFAULT_FACT_LIMIT, ge=1, le=MAX_FACT_LIMIT),
    category: Optional[str] = Query(None),
    min_trust: float = Query(0.0, ge=0.0, le=1.0),
    search: Optional[str] = Query(None),
) -> Dict[str, Any]:
    return _holographic_payload(limit=limit, category=category or None, min_trust=min_trust, search=search or None)


@router.get("/mem0")
async def mem0(
    limit: int = Query(DEFAULT_MEM0_LIMIT, ge=1, le=MAX_MEM0_LIMIT),
    search: Optional[str] = Query(None),
) -> Dict[str, Any]:
    return _mem0_payload(limit=limit, search=search or None)


@router.get("/honcho")
async def honcho(
    limit: int = Query(DEFAULT_HONCHO_LIMIT, ge=1, le=MAX_HONCHO_LIMIT),
    search: Optional[str] = Query(None),
) -> Dict[str, Any]:
    return _honcho_payload(limit=limit, search=search or None)


@router.get("/hindsight")
async def hindsight() -> Dict[str, Any]:
    return _hindsight_payload(mode="status")


@router.get("/hindsight/contents")
async def hindsight_contents(
    limit: int = Query(DEFAULT_HINDSIGHT_LIMIT, ge=1, le=MAX_HINDSIGHT_LIMIT),
    search: Optional[str] = Query(None),
) -> Dict[str, Any]:
    return _hindsight_contents_payload(limit=limit, search=search or None)


@router.get("/hindsight/recall")
async def hindsight_recall(
    query: str = Query(...),
    limit: int = Query(DEFAULT_HINDSIGHT_LIMIT, ge=1, le=MAX_HINDSIGHT_LIMIT),
) -> Dict[str, Any]:
    return _hindsight_payload(query=query, limit=limit, mode="recall")


@router.get("/hindsight/reflect")
async def hindsight_reflect(
    query: str = Query(...),
) -> Dict[str, Any]:
    return _hindsight_payload(query=query, limit=1, mode="reflect")


@router.get("/snapshot")
async def snapshot(
    limit: int = Query(DEFAULT_FACT_LIMIT, ge=1, le=MAX_FACT_LIMIT),
    category: Optional[str] = Query(None),
    min_trust: float = Query(0.0, ge=0.0, le=1.0),
    search: Optional[str] = Query(None),
) -> Dict[str, Any]:
    config = _read_yaml(_hermes_home() / "config.yaml")
    return {
        "plugin": "hermes-memory-ui",
        "version": "0.4.6",
        "mode": "read-only",
        "builtin": _builtin_payload(config),
        "holographic": _holographic_payload(
            config,
            limit=limit,
            category=category or None,
            min_trust=min_trust,
            search=search or None,
        ),
        "mem0": _mem0_payload(config, limit=limit, search=search or None),
        "honcho": _honcho_payload(config, limit=limit, search=search or None),
        "hindsight": _hindsight_payload(config, mode="status"),
        "generated_at": time.time(),
    }
