import os, io, hashlib
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.eda_agent.state import dataset_id_from_bytes, DatasetMemory
from src.eda_agent.agent_codegen import plan_and_execute

load_dotenv()

st.set_page_config(page_title="EDA Agent – Explore seus dados", layout="wide")
st.title("📊 EDA Agent – Explore seus dados")

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

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "df" not in st.session_state:
    st.session_state.df = None

# =========================
# Leitura de CSV otimizada
# =========================
SAMPLE_SIZE = 65536  # 64KB para detecção rápida

def _fast_hash(b: bytes) -> str:
    # hash rápido para cache interno, se quiser usar
    return hashlib.blake2b(b, digest_size=16).hexdigest()

def detect_encoding_sample(content: bytes) -> str:
    """
    Detecta encoding usando apenas um sample para não ficar lento.
    Tenta chardet se disponível; senão: utf-8 -> cp1252.
    """
    sample = content[:SAMPLE_SIZE]
    try:
        import chardet  # opcional
        res = chardet.detect(sample)
        enc = (res.get("encoding") or "").lower()
        if enc in ("utf-8", "utf_8", "utf8"):
            return "utf-8"
        if enc in ("cp1252", "windows-1252", "latin-1", "iso-8859-1", "latin1"):
            return "cp1252"
        # fallback razoável
        return "utf-8"
    except Exception:
        try:
            sample.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "cp1252"

def detect_separator_sample(content: bytes, encoding: str) -> str:
    """
    Decide entre ',' e ';' contando ocorrências no sample (sem engine python).
    """
    text = content[:SAMPLE_SIZE].decode(encoding, errors="ignore").strip()
    return ";" if text.count(";") > text.count(",") else ","

@st.cache_data(show_spinner=False)
def read_csv_fast(file_bytes: bytes) -> pd.DataFrame:
    """
    Caminho rápido: engine 'c' com sep fixo e encoding detectado por sample.
    Fallback: engine 'python' apenas se necessário.
    """
    enc = detect_encoding_sample(file_bytes)
    sep = detect_separator_sample(file_bytes, enc)

    bio = io.BytesIO(file_bytes)

    # Caminho feliz: engine C (mais rápido). Evite on_bad_lines aqui.
    try:
        return pd.read_csv(bio, sep=sep, encoding=enc, engine="c")
    except Exception:
        # Tenta engine C com encoding alternativo
        alt_enc = "cp1252" if enc == "utf-8" else "utf-8"
        bio.seek(0)
        try:
            return pd.read_csv(bio, sep=sep, encoding=alt_enc, engine="c")
        except Exception:
            # Fallback: engine python (aceita on_bad_lines)
            bio.seek(0)
            try:
                return pd.read_csv(bio, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
            except Exception:
                bio.seek(0)
                return pd.read_csv(bio, sep=sep, encoding=alt_enc, engine="python", on_bad_lines="skip")

# =========================
# Upload e parsing
# =========================
if up:
    content = up.read()
    st.session_state.dataset_id = dataset_id_from_bytes(content)
    try:
        df_raw = read_csv_fast(content)  # usa cache por conteúdo
        df_clean = df_raw.dropna(how="all").dropna(axis=1, how="all")
        df_clean.columns = df_clean.columns.str.strip()
        st.session_state.df = df_clean
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

    n_total = len(df.index)
    n_cols = len(df.columns)
    n_show = min(n_total, 1000)

    if n_total == 0:
        st.warning("O CSV foi lido, mas não há linhas (0 registros). Verifique separador e encoding.")
    else:
        st.caption(f"{n_total} linhas × {n_cols} colunas • Mostrando {n_show} linha(s)")
        st.dataframe(df.head(n_show), use_container_width=True)

    mem = DatasetMemory.load(dataset_id)

    # Botão para limpar conclusões
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

    if st.button("Perguntar", type="primary") and question:
        with st.spinner("Gerando código e executando..."):
            out = plan_and_execute(
                question, df, mem,
                llm_model=llm_model,
                temperature=temperature,
                enable_critic=enable_critic,
            )
        st.markdown(out.get("text") or "")
        if out.get("stdout"):
            with st.expander("Saída (stdout) do código"):
                st.code(out["stdout"])
        for img_bytes in out.get("images", []):
            st.image(img_bytes)
        with st.expander("Código gerado"):
            st.code(out.get("code") or "")
        if enable_critic and (out.get("critic") or "").strip():
            st.divider()
            st.subheader("🧠 Conclusões críticas")
            st.markdown(out["critic"])
else:
    st.info("Carregue um CSV para começar.")
