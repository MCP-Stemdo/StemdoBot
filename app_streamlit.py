import streamlit as st
import requests
import base64

# Configurar la URL del backend (FastAPI)
BACKEND_URL = "http://127.0.0.1:8000"

# Función para hacer una pregunta al modelo
def ask_question(question):
    response = requests.post(f"{BACKEND_URL}/ask_question", data={"question": question})
    return response

# Función para subir documentos al backend
def upload_document(file):
    files = {"file": file}
    response = requests.post(f"{BACKEND_URL}/upload_document", files=files)
    return response

# Función para listar documentos
def list_documents():
    response = requests.get(f"{BACKEND_URL}/list_documents/")
    return response

# Función para eliminar documentos
def delete_document(document_id):
    response = requests.delete(f"{BACKEND_URL}/delete_document/{document_id}/")
    return response

# Función para entrenar el modelo
def train_model():
    response = requests.post(f"{BACKEND_URL}/train_model")
    return response

# Función para crear el pie de página
def footer():
    st.markdown("""<style>footer { position: fixed; bottom: 0; left: 0; width: 100%; height: 70px; background-color: #4c946b; text-align: center; padding: 10px; color: white; font-size: 14px; display: flex; justify-content: right; align-items: center; z-index: 10; } footer img { margin-right: 10px; height: 60px; } footer p { margin: 0; }</style>""", unsafe_allow_html=True)

    footer_image_path = "data/img/favicon-32x32.png"
    st.markdown(f"""<footer><img src="data:image/png;base64,{get_image_base64(footer_image_path)}" alt="Logo"><p>Desarrollado por Stemdo © 2024 | <a href="mailto:people@stemdo.io" style="color: white; text-decoration: underline;"> CONTACTO </a></p></footer>""", unsafe_allow_html=True)

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Función principal del frontend
def main():
    st.set_page_config(page_title="StemdoBot 🤖", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

    # Cargar la imagen GIF como base64
    sidebar_image_path = "data/img/81mg.gif"
    sidebar_image_base64 = get_image_base64(sidebar_image_path)

    # Estilos personalizados para el sidebar y botones
    st.markdown(f"""
        <style>
        #MainMenu {{visibility: hidden;}} 
        footer {{visibility: visible;}} 
        .stSidebar {{
            background-image: url('data:image/gif;base64,{sidebar_image_base64}'); 
            background-size: fit-content; 
            background-repeat: no-repeat;
            background-position: bottom;
            background-color: #020202;
        }}
        .stButton > button {{
            background-color: #4c946b;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }}
        .stButton > button:hover {{
            background-color: #020202;
            color: white;
        }}
        .document-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0; /* Espaciado interno para los elementos */
        }}
        .document-item button {{
            background-color: transparent; /* Color de fondo del botón de eliminar */
            border: none; /* Sin borde */
            color: #ff4c4c; /* Color del texto del botón de eliminar */
            cursor: pointer; /* Cambia el cursor al pasar el mouse */
            font-size: 16px; /* Tamaño de fuente */
            margin-left: 10px; /* Espacio entre el texto y el botón */
        }}
        </style>
    """, unsafe_allow_html=True)

    st.title("StemdoBot 🤖")

    # Crear un menú de navegación
    menu = ["Stemdoer", "Administrador"]
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    # Sección para los usuarios tipo Stemdoer
    if choice == "Stemdoer":
        st.header("Haz tu pregunta")

        # Cuadro de texto para la pregunta del usuario
        question = st.text_input("Pregunta:")

        # Botón para obtener respuesta
        if st.button("Obtener respuesta"):
            if question.strip() == "":
                st.error("Por favor, ingresa una pregunta válida.", icon="🚨")
            else:
                try:
                    with st.spinner("Procesando..."):
                        # Hacer la solicitud POST al backend con la pregunta
                        response = ask_question(question)

                        if response.status_code == 200:
                            answer = response.json().get("answer")
                            st.success(f"Respuesta del modelo: {answer}")
                        else:
                            st.error(f"Error del servidor: {response.status_code}", icon="🚨")
                except Exception as e:
                    st.error(f"Error al conectar con el servidor: {e}", icon="🚨")

    # Sección para el administrador
    elif choice == "Administrador":
        st.header("Panel de administración")

        # Subir documentos
        st.subheader("Subir documento")
        uploaded_file = st.file_uploader("Selecciona un archivo", type=["pdf", "txt"], label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("Subir documento"):
                with st.spinner("Subiendo documento..."):
                    response = upload_document(uploaded_file)

                    if response.status_code == 200:
                        result = response.json()
                        message = result.get("message")
                        st.success(f"{message}")
                    else:
                        st.error(f"Error al subir documento: {response.status_code}")

        # Listar documentos existentes
        st.subheader("Documentos subidos")
        documents_response = list_documents()
        if documents_response.status_code == 200:
            documents = documents_response.json().get("documents", [])
            if documents:
                for doc in documents:
                    # Asumiendo que `doc` es un diccionario con 'id' y 'name'
                    doc_id = doc['id']  # Cambiado para usar clave 'id'
                    doc_name = doc.get('original_filename')  # Cambiado para usar clave 'name'
                    if doc_name is None:
                        st.error(f"El documento con id {doc['id']} no tiene un nombre asociado")
                        continue
                    # Mostrar el nombre del documento y el botón de eliminar
                    st.markdown(f'<div class="document-item">{doc_name}<button onclick="deleteDocument({doc_id})">╳</button></div>', unsafe_allow_html=True)
            else:
                st.write("No hay documentos subidos.")
        else:
            st.error(f"Error al listar documentos: {documents_response.status_code}")

        # Entrenar el modelo
        st.subheader("Entrenar el modelo")
        if st.button("Entrenar modelo"):
            with st.spinner("Entrenando el modelo..."):
                response = train_model()
                if response.status_code == 200:
                    st.success("Modelo entrenado correctamente")
                else:
                    st.error(f"Error al entrenar modelo: {response.status_code}")

    # Insertar el pie de página
    footer()

if __name__ == "__main__":
    main()
