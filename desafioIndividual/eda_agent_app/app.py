import os, io, hashlib
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.eda_agent.state import dataset_id_from_bytes, DatasetMemory
from src.eda_agent.agents.codegen_agent import generate_and_execute, build_schema_hint, _format_history
from src.eda_agent.agents.critic_agent import run_critic
from src.eda_agent.agents.summary_agent import summarize_memory

load_dotenv()

st.set_page_config(page_title="EDA Agent – Explore seus dados", layout="wide")
st.title("📊 EDA Agent – Explore seus dados")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Upload do CSV")
    up = st.file_uploader("Escolha um arquivo .csv", type=["csv"])
    st.divider()
    st.header("LLM")
    llm_model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
    st.caption("Defina OPENAI_API_KEY no ambiente.")
    st.divider()
    st.header("Conclusões críticas")
    enable_critic = st.checkbox("Gerar conclusões críticas (pós-execução)", value=True)
    st.divider()
    st.header("Memória")
    show_memory = st.checkbox("Mostrar conclusões salvas", value=True)
    st.caption("Use o botão abaixo para um resumo sem executar código.")
    summarize_now = st.button("🧠 Resumir conclusões (sem executar código)")

# =========================
# Estado
# =========================
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "df" not in st.session_state:
    st.session_state.df = None

# =========================
# Funções auxiliares de leitura
# =========================
SAMPLE_SIZE = 65536  # 64KB

def _fast_hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()

def detect_encoding_sample(content: bytes) -> str:
    sample = content[:SAMPLE_SIZE]
    try:
        import chardet
        res = chardet.detect(sample)
        enc = (res.get("encoding") or "").lower()
        if enc.startswith("utf"):
            return "utf-8"
        if "1252" in enc or "8859" in enc or "latin" in enc:
            return "cp1252"
        return "utf-8"
    except Exception:
        try:
            sample.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"

def detect_separator_sample(content: bytes, encoding: str) -> str:
    text = content[:SAMPLE_SIZE].decode(encoding, errors="ignore").strip()
    return ";" if text.count(";") > text.count(",") else ","

@st.cache_data(show_spinner=False)
def read_csv_fast(file_bytes: bytes) -> pd.DataFrame:
    enc = detect_encoding_sample(file_bytes)
    sep = detect_separator_sample(file_bytes, enc)
    bio = io.BytesIO(file_bytes)
    try:
        return pd.read_csv(bio, sep=sep, encoding=enc, engine="c")
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")

# =========================
# Upload
# =========================
if up:
    content = up.read()
    st.session_state.dataset_id = dataset_id_from_bytes(content)
    try:
        df = read_csv_fast(content)
        # limpeza extra
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df.columns = df.columns.str.strip()
        st.session_state.df = df
        st.success(f"Dataset carregado. ID: {st.session_state.dataset_id}")
    except Exception as e:
        st.error(f"Falha ao ler o CSV: {e}")
        st.session_state.df = None

df = st.session_state.df
dataset_id = st.session_state.dataset_id

# =========================
# UI principal
# =========================
if df is not None:
    st.write("Amostra dos dados:")
    n_total, n_cols = df.shape
    n_show = min(n_total, 1000)
    st.caption(f"{n_total} linhas × {n_cols} colunas • Mostrando {n_show} linha(s)")
    st.dataframe(df.head(n_show), width='stretch')

    mem = DatasetMemory.load(dataset_id)

    if st.button("🧹 Limpar conclusões deste dataset"):
        mem.conclusions = []
        mem.save()
        st.success("Conclusões apagadas.")

    if show_memory:
        with st.expander("Conclusões (memória por dataset)"):
            if mem.conclusions:
                for i, c in enumerate(mem.conclusions, 1):
                    st.markdown(f"{i}. {c}")
            else:
                st.caption("Nenhuma conclusão armazenada ainda.")
        with st.expander("Histórico (últimos 5 turnos)"):
            for t in mem.recent_turns(5):
                q = t.get("question","")
                a = t.get("result_text","")
                st.markdown(f"**Q:** {q}\n\n**Conclusão:** {a}")

    st.divider()
    st.subheader("Faça uma pergunta")
    question = st.text_input("Ex.: 'Média e histograma da coluna idade' ou 'Correlação entre renda e gasto'")

    # resumo sem execução
    wants_summary = False
    if question:
        qlow = question.strip().lower()
        wants_summary = any(k in qlow for k in [
            "resumo das conclusões", "resumir conclusões", "conclusão geral", "síntese", "sumário"
        ])

    if summarize_now or wants_summary:
        with st.spinner("Gerando resumo executivo (sem executar código)..."):
            summary = summarize_memory(mem, llm_model=llm_model, temperature=0.2)
        st.subheader("🧠 Síntese do que já foi aprendido até agora")
        st.markdown(summary or "_Sem conteúdo para resumir ainda._")

    elif st.button("Perguntar", type="primary") and question:
        with st.spinner("Gerando código e executando..."):
            out = generate_and_execute(
                question, df, mem,
                llm_model=llm_model,
                temperature=temperature,
            )
        st.markdown(out.get("text") or "")
        if out.get("stdout"):
            with st.expander("Saída (stdout) do código"):
                st.code(out["stdout"])
        for img_bytes in out.get("images", []):
            st.image(img_bytes)
        with st.expander("Código gerado"):
            st.code(out.get("code") or "")

        if enable_critic:
            with st.spinner("Gerando conclusões críticas..."):
                hint = build_schema_hint(df)
                history_snippet = _format_history(mem.recent_turns(k=5))
                critic_text = run_critic(
                    question=question,
                    history_snippet=history_snippet,
                    schema_hint=hint,
                    result_text=out.get("text",""),
                    stdout_tail=out.get("stdout","") or "",
                    llm_model=llm_model,
                    temperature=0.2,
                )
                for line in critic_text.splitlines():
                    s = line.strip()
                    if s.startswith(("-", "•")) and len(s) > 2:
                        mem.add_conclusion(s.lstrip("-• ").strip())
            st.divider()
            st.subheader("🧠 Conclusões críticas")
            st.markdown(critic_text)

else:
    st.info("Carregue um CSV para começar.")
