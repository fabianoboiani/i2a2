from __future__ import annotations
import os, json, hashlib, time
from dataclasses import dataclass, field
from typing import Dict, Any, List

CACHE_DIR = os.environ.get("EDA_AGENT_CACHE_DIR", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def dataset_id_from_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

@dataclass
class DatasetMemory:
    dataset_id: str
    conclusions: List[str] = field(default_factory=list)
    summaries: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join(CACHE_DIR, f"{self.dataset_id}.json")

    @classmethod
    def load(cls, dataset_id: str) -> "DatasetMemory":
        path = os.path.join(CACHE_DIR, f"{dataset_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return cls(
                dataset_id=dataset_id,
                conclusions=d.get("conclusions", []),
                summaries=d.get("summaries", {}),
                chat_history=d.get("chat_history", []),
            )
        return cls(dataset_id=dataset_id)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conclusions": self.conclusions,
                    "summaries": self.summaries,
                    "chat_history": self.chat_history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add_conclusion(self, text: str) -> None:
        if text and text.strip():
            self.conclusions.append(text.strip())
            self.save()

    # Registrar um turno de conversa
    def add_turn(self, question: str, result_text: str, code: str) -> None:
        self.chat_history.append({
            "ts": int(time.time()),
            "question": (question or "").strip(),
            "result_text": (result_text or "").strip(),
            # armazenar preview do código para não inflar o arquivo
            "code_preview": (code or "").strip()[:2000],
        })
        # manter somente os últimos 50 turnos
        self.chat_history = self.chat_history[-50:]
        self.save()

    # Recuperar últimos N turnos
    def recent_turns(self, k: int = 5) -> List[Dict[str, Any]]:
        return self.chat_history[-k:]
