# 🧠 EDA Agent

Agente de IA para **análise exploratória de dados (EDA)** que interpreta perguntas em linguagem natural e gera **código Python automaticamente** para responder, exibindo resultados e gráficos.

---

## ⚙️ Como funciona
- A LLM interpreta a pergunta do usuário.  
- Gera o código Python (usando `pandas`, `numpy`, `matplotlib`, etc.).  
- O código é executado em **sandbox seguro**.  
- O resultado e os gráficos são exibidos via **Streamlit**.

---

## 🚀 Execução

```bash
poetry install
poetry run streamlit run app.py
