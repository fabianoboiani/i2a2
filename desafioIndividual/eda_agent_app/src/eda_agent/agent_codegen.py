from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .state import DatasetMemory
from .executor import run_generated_code

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
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }

def _format_history(turns: List[Dict[str, Any]]) -> str:
    """
    Gera um resumo compacto (pergunta -> conclusão) para orientar o codegen.
    Não inclui dados brutos; apenas contexto textual útil.
    """
    if not turns:
        return "Nenhum histórico relevante."
    lines = []
    for t in turns:
        q = (t.get("question") or "").strip()
        a = (t.get("result_text") or "").strip()
        if q or a:
            lines.append(f"- Pergunta: {q}\n  Conclusão: {a}")
    return "\n".join(lines) if lines else "Nenhum histórico relevante."

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
- Se houver ambiguidade no contexto, sinalize-a em Limitações.
- Responda em português.
"""

def _critic_summarize(llm: ChatOpenAI, *, question: str, history_snippet: str, schema_hint: dict,
                      result_text: str, stdout_tail: str) -> str:
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

def plan_and_execute(question: str, df: pd.DataFrame, memory: DatasetMemory,
                     llm_model: str="gpt-4o-mini", temperature: float=0.0,
                     enable_critic: bool = True) -> Dict[str, Any]:
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
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
- Se a pergunta referir-se a algo previamente analisado (ex.: "a mesma coluna de antes"), infira de forma conservadora a partir do HISTÓRICO; se houver ambiguidade, mencione-a no RESULT_TEXT e adote a decisão mais prudente (ex.: peça o nome da coluna na próxima interação).
- Gere gráficos quando fizer sentido.
"""
    msg = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    out = llm.invoke(msg).content

    code = out
    if "```" in out:
        import re
        m = re.search(r"```(?:python)?\s*(.*?)```", out, re.S)
        if m:
            code = m.group(1).strip()

    import numpy as np
    import matplotlib.pyplot as plt
    exec_result = run_generated_code(code, extra_globals={"pd": pd, "np": np, "plt": plt, "df": df})

    if exec_result.get("text"):
        memory.add_conclusion(exec_result["text"])
    # registrar turno (pergunta, conclusão e preview do código)
    memory.add_turn(
        question=question,
        result_text=exec_result.get("text") or "",
        code=code or ""
    )

    critic_text = ""
    if enable_critic:
        critic_llm = ChatOpenAI(model=llm_model, temperature=0.2)
        stdout_tail = exec_result.get("stdout", "") or ""
        critic_text = _critic_summarize(
            critic_llm,
            question=question,
            history_snippet=history_snippet,
            schema_hint=hint,
            result_text=exec_result.get("text",""),
            stdout_tail=stdout_tail,
        )
        # Persistir bullets do crítico como conclusões adicionais
        for line in critic_text.splitlines():
            s = line.strip()
            if s.startswith(("-", "•")) and len(s) > 2:
                memory.add_conclusion(s.lstrip("-• ").strip())

    return {
        "code": code,
        "text": exec_result.get("text", ""),
        "stdout": exec_result.get("stdout", ""),
        "images": exec_result.get("images", []),
        "critic": critic_text,
    }
