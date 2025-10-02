from __future__ import annotations
from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from .base import build_llm

CRITIC_SYSTEM = """Você é um analista de dados sênior. Sua tarefa é produzir CONCLUSÕES CRÍTICAS claras e acionáveis
com base em:
- a pergunta do usuário,
- o histórico recente (pergunta → conclusão),
- o schema do dataset,
- o RESULT_TEXT e um trecho do stdout do código executado pelo agente.

Regras:
- Seja objetivo e opinativo quando apropriado.
- Evite repetir números que não agreguem; explique o que eles significam.
- Traga sempre 3 seções, nesta ordem:
  1) Conclusões (3–6 bullets; cada bullet deve ter evidência curta ou racional),
  2) Limitações (2–4 bullets),
  3) Próximos passos (3–5 bullets, práticos e ordenados por impacto).
- Se houver ambiguidade, sinalize-a em Limitações.
- Responda em português.
"""

def run_critic(*, question: str, history_snippet: str, schema_hint: dict,
               result_text: str, stdout_tail: str,
               llm_model: str="gpt-4o-mini", temperature: float=0.2) -> str:
    llm = build_llm(model=llm_model, temperature=temperature)
    prompt = f"""
PERGUNTA: {question}

HISTÓRICO (recente):
{history_snippet}

SCHEMA (JSON):
{schema_hint}

RESULTADO DO CÓDIGO (RESULT_TEXT):
{result_text}

TRECHO DO STDOUT (até 1200 chars):
{stdout_tail[:1200]}
"""
    msgs = [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=prompt)]
    return llm.invoke(msgs).content.strip()
