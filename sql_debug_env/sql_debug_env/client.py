"""HTTP client for the SQL Debug & Optimize RL Environment."""

from __future__ import annotations
import httpx
from typing import Optional
from .models import SQLDebugAction, SQLDebugObservation, SQLDebugState


class SQLDebugEnv:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "SQLDebugEnv":
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def reset(self, seed: Optional[int] = None) -> SQLDebugObservation:
        resp = self._post("/reset", {"seed": seed})
        return SQLDebugObservation(**resp)

    def step(self, action: SQLDebugAction) -> SQLDebugObservation:
        resp = self._post("/step", {"fixed_query": action.fixed_query, "explanation": action.explanation})
        return SQLDebugObservation(**resp)

    def get_state(self) -> SQLDebugState:
        r = self._get_client().get("/state")
        r.raise_for_status()
        return SQLDebugState(**r.json())

    def health(self) -> dict:
        r = self._get_client().get("/health")
        r.raise_for_status()
        return r.json()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _post(self, path: str, payload: dict) -> dict:
        r = self._get_client().post(path, json=payload)
        r.raise_for_status()
        return r.json()
