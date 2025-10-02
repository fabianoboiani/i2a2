from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.schema import HumanMessage, SystemMessage
from ..state import DatasetMemory
from ..executor import run_generated_code
from .base import build_llm

SYSTEM = """Você é um engenheiro de dados que GERA CÓDIGO PYTHON para responder perguntas sobre um DataFrame 'df' (pandas).
REGRAS OBRIGATÓRIAS:
- Gere apenas código Python válido, sem explicações fora do bloco de código.
- NÃO use 'import' no código. As bibliotecas já estão disponíveis como: pandas 'pd', numpy 'np', matplotlib.pyplot 'plt'.
- O DataFrame já está disponível como variável 'df'.
- Se criar gráficos, use matplotlib (plt). Não mostre os gráficos; apenas crie-os (a aplicação host irá capturar as figuras).
- Coloque a resposta textual em uma variável: RESULT_TEXT = "...".
- Não leia/escreva arquivos. Não acesse rede. Não use funções perigosas.
- Valide a existência e os tipos das colunas antes de operar. Seja robusto a NaNs.
"""

def build_schema_hint(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}

def _format_history(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return "Nenhum histórico relevante."
    lines = []
    for t in turns:
        q = (t.get("question") or "").strip()
        a = (t.get("result_text") or "").strip()
        if q or a:
            lines.append(f"- Pergunta: {q}\n  Conclusão: {a}")
    return "\n".join(lines) if lines else "Nenhum histórico relevante."

def generate_and_execute(question: str, df: pd.DataFrame, memory: DatasetMemory,
                         llm_model: str="gpt-4o-mini", temperature: float=0.0,
                         enable_critic: bool = False) -> Dict[str, Any]:
    """
    Agente Codegen: gera código Python, executa em sandbox e persiste resultado/turno.
    (Se quiser o crítico, invoque o critic_agent a partir do app após esse retorno.)
    """
    llm = build_llm(model=llm_model, temperature=temperature)
    hint = build_schema_hint(df)
    history_snippet = _format_history(memory.recent_turns(k=5))

    prompt = f"""
PERGUNTA ATUAL: {question}

HISTÓRICO RECENTE (pergunta → conclusão):
{history_snippet}

SCHEMA (JSON): {hint}

Gere APENAS um snippet Python que, quando executado, produza a resposta para a pergunta atual.
Regras:
- Use 'df' (pandas), 'pd', 'np', 'plt'. Não use 'import'.
- Defina 'RESULT_TEXT' com a conclusão principal, integrando o contexto do histórico quando fizer sentido.
- Se a pergunta referir-se a algo previamente analisado, infira de forma conservadora a partir do HISTÓRICO; se houver ambiguidade, mencione-a em RESULT_TEXT.
- Gere gráficos quando fizer sentido.
"""
    msgs = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    out = llm.invoke(msgs).content

    code = out
    if "```" in out:
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", out, re.S)
        if m:
            code = m.group(1).strip()

    exec_result = run_generated_code(code, extra_globals={"pd": pd, "np": np, "plt": plt, "df": df})

    if exec_result.get("text"):
        memory.add_conclusion(exec_result["text"])
    memory.add_turn(question=question, result_text=exec_result.get("text") or "", code=code or "")

    return {
        "code": code,
        "text": exec_result.get("text", ""),
        "stdout": exec_result.get("stdout", ""),
        "images": exec_result.get("images", []),
    }
