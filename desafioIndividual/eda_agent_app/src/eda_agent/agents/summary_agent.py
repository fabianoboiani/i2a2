from __future__ import annotations
from typing import Dict, Any, List
from langchain.schema import HumanMessage, SystemMessage
from ..state import DatasetMemory
from .base import build_llm

SUMMARY_SYSTEM = """Você é um analista de dados sênior. Sua tarefa é produzir um RESUMO EXECUTIVO conciso
com base APENAS no histórico (pergunta → conclusão) e nas conclusões críticas já salvas.
Não execute código, não invente colunas nem valores numéricos que não estejam no texto.

Formato (em português):
1) Principais insights (3–6 bullets)
2) Limitações (2–4 bullets)
3) Próximos passos (3–5 bullets)
"""

def summarize_memory(memory: DatasetMemory, llm_model: str="gpt-4o-mini",
                     temperature: float=0.2, max_turns: int = 8, max_conclusions: int = 30) -> str:
    llm = build_llm(model=llm_model, temperature=temperature)

    turns = memory.recent_turns(k=max_turns)
    hist = []
    for t in turns:
        q = (t.get("question") or "").strip()
        a = (t.get("result_text") or "").strip()
        if q or a:
            hist.append(f"- Pergunta: {q}\n  Conclusão: {a}")
    history_snippet = "\n".join(hist) if hist else "Sem histórico recente."

    cons = memory.conclusions[-max_conclusions:] if memory.conclusions else []
    conclusions_snippet = "\n".join(f"- {c}" for c in cons) if cons else "Sem conclusões salvas."

    prompt = f"""
HISTÓRICO (recente, pergunta → conclusão):
{history_snippet}

CONCLUSÕES SALVAS (amostra):
{conclusions_snippet}
"""
    msgs = [SystemMessage(content=SUMMARY_SYSTEM), HumanMessage(content=prompt)]
    return llm.invoke(msgs).content.strip()
