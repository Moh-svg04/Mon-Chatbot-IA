import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration de la page
st.set_page_config(page_title="Mon Chatbot IA", page_icon="🤖")
st.title("🤖 Mon Premier Chatbot IA")
st.markdown("Pose-moi une question et je te répondrai en utilisant un modèle de Deep Learning !")

# 1. Chargement du modèle (Blenderbot de Facebook)
@st.cache_resource # Pour ne pas recharger le modèle à chaque clic
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 2. Gestion de l'historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Zone de saisie utilisateur
if prompt := st.chat_input("Dis-moi quelque chose..."):
    # Afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse de l'IA
    with st.chat_message("assistant"):
        # Préparation du texte pour le modèle
        inputs = tokenizer(prompt, return_tensors="pt")
        res = model.generate(**inputs)
        response = tokenizer.decode(res[0], skip_special_tokens=True)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
