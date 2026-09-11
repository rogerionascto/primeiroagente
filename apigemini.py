import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# ==========================================
# 1. CONFIGURAÇÃO INICIAL E AUTENTICAÇÃO
# ==========================================

st.set_page_config(page_title="Agente Groq", page_icon="🤖")
st.title("🤖 Chatbot com Groq")

pasta_do_projeto = Path(__file__).resolve().parent
load_dotenv(dotenv_path=pasta_do_projeto / ".env")

chave_api = os.getenv("GROQ_API_KEY")

if not chave_api:
    st.error("Erro: Adicione a chave GROQ_API_KEY no arquivo .env para continuar.")
    st.stop()

cliente_groq = Groq(api_key=chave_api)

# Instrução fixa de comportamento (System Prompt)
INSTRUCAO_SISTEMA = {
    "role": "system",
    "content": "Você é um corinthiano roxo com sotaque clássico de quebrada/Itaquera, bem maloqueiro, gente boa e prestativo.",
}

# ==========================================
# 2. FUNÇÃO QUE CONVERSA COM A IA
# ==========================================

def consultar_groq(historico_de_mensagens):
    """Monta a lista com a personalidade e faz a chamada via stream."""
    mensagens_completas = [INSTRUCAO_SISTEMA] + historico_de_mensagens

    return cliente_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=mensagens_completas,
        stream=True,
        temperature=0.7,
        max_completion_tokens=400,
        presence_penalty=0.6,
        frequency_penalty=0.3,
    )

def filtrar_texto(stream):
    """Extrai apenas o texto final, ignorando chunks de raciocínio e metadados."""
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            conteudo = chunk.choices[0].delta.content
            if conteudo:
                yield conteudo

# ==========================================
# 3. INTERFACE E CONTROLE DO CHAT
# ==========================================

if "historico" not in st.session_state:
    st.session_state.historico = []

# Exibe na tela as mensagens trocadas até o momento
for mensagem in st.session_state.historico:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

# Caixa para entrada do usuário
pergunta_usuario = st.chat_input("Diga aí, meu parça...")

if pergunta_usuario:
    # 1. Registra e renderiza a pergunta do usuário
    st.session_state.historico.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    # 2. Gera a resposta filtrada em tempo real
    with st.chat_message("assistant"):
        resposta_em_stream = consultar_groq(st.session_state.historico)
        texto_final = st.write_stream(filtrar_texto(resposta_em_stream))

    # 3. Salva a resposta do assistente no histórico
    st.session_state.historico.append({"role": "assistant", "content": texto_final})