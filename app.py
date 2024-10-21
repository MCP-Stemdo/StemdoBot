from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import numpy as np

app = FastAPI()

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Permitir CORS para que el frontend pueda comunicarse con el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de embeddings Hugging Face localmente
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

ALLOWED_EXTENSIONS = {"txt", "pdf"}

# Función para sanitizar nombre de colección
def sanitize_collection_name(collection_name):
    collection_name = collection_name.strip('_').replace(" ", "_").replace("-", "_")
    collection_name = collection_name.strip('_-')
    if len(collection_name) < 3:
        collection_name += "001"
    elif len(collection_name) > 63:
        collection_name = collection_name[:63]
    valid_name = ''.join(c for c in collection_name if c.isalnum() or c in ['_', '-'])
    return valid_name

# Verificar tipos de archivos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Crear embedding desde el documento
def create_embedding(file_path, file_type, original_filename):
    collection_name = "documents" 
    try:
        # Cargar documentos según el tipo de archivo
        if file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Cargar y dividir documentos
        documents = loader.load()
        logging.info("Documents loaded: %s", [doc.page_content for doc in documents])
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=15)
        texts = text_splitter.split_documents(documents)

        # Verificar que se han generado textos
        if not texts:
            logging.warning("No text was split from the documents.")
            raise ValueError("No text was split from the documents.")

        # Generar embeddings con el modelo local
        embeddings = embedding_model.encode([doc.page_content for doc in texts])
        logging.info("Embeddings shape: %s", embeddings.shape)

        # Crear cliente Chroma y colección
        client = chromadb.Client()

        # Comprobar si la colección "documents" ya existe
        if "documents" in [col.name for col in client.list_collections()]:
            collection = client.get_collection("documents")
            logging.info("Using existing collection: %s", "documents")
        else:
            collection = client.create_collection(name="documents")
            logging.info("Created new collection: %s", "documents")

        # Añadir documentos a la colección
        for i, text in enumerate(texts):
            collection.add(
                documents=[text.page_content],
                embeddings=[embeddings[i].tolist()],
                metadatas=[{"page": text.metadata.get("page_number", 1)}],
                ids=[f"{original_filename}_{i + 1}"]  
            )

        return collection

    except Exception as e:
        logging.error("Error during embedding creation: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# Endpoint para subir documentos
@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Tipo de archivo no permitido. Solo se permiten archivos .txt y .pdf.")

        if not os.path.exists("data"):
            os.makedirs("data")

        file_path = f"data/{file.filename}"
        
        if os.path.exists(file_path):
            os.remove(file_path)
            message = f"El archivo {file.filename} ya existía. Ha sido reemplazado."
        else:
            message = f"El archivo {file.filename} se ha subido correctamente."
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        file_type = file.filename.split(".")[-1]
        create_embedding(file_path, file_type, file.filename) 

        return {"message": message, "status": 200}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo archivo: {e}")

# Endpoint para listar archivos
@app.get("/list_documents/")
async def list_documents():
    try:
        client = chromadb.Client()
        collections = client.list_collections()

        # Verificar si la colección "documents" existe
        if "documents" not in [col.name for col in collections]:
            return {"message": "La colección 'documents' no existe. Sube un documento primero."}

        collection = client.get_collection("documents")  
        documents = collection.get()

        # Formatear la respuesta para incluir el nombre del archivo en lugar de solo IDs
        formatted_documents = [{"id": doc[0], "original_filename": doc[0].rsplit('_', 1)[0]} for doc in documents['documents']]
        
        return {"documents": formatted_documents, "status": 200}
    
    except Exception as e:
        logging.error("Error al listar documentos: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error al listar documentos: {e}")

# Endpoint para eliminar documentos
@app.delete("/delete_document/{document_id}/")
async def delete_document(document_id: str):
    try:
        client = chromadb.Client()
        collection = client.get_collection("documents")  
        collection.delete(ids=[document_id])
        return {"message": f"Documento {document_id} eliminado correctamente.", "status": 200}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar documento: {e}")

# Endpoint para hacer preguntas
@app.post("/ask_question/")
async def ask_question(question: str = Form(...)):
    try:
        question_embedding = embedding_model.encode([question])[0]
        client = chromadb.Client()
        
        collections = client.list_collections()
        if not collections:
            raise HTTPException(status_code=400, detail="No se ha encontrado ninguna colección. Sube un documento primero.")

        collection = client.get_collection("documents")  

        results = collection.query(query_embeddings=[question_embedding], n_results=1)

        if not results or not results['documents']:
            logging.info("No results found for the question: %s", question)
            raise HTTPException(status_code=404, detail="No se encontraron respuestas para la pregunta. Intenta con otra pregunta.")

        answer = results['documents'][0][0]
        return {"answer": answer}
    
    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {str(e)}")

# Entrenar el modelo (placeholder)
@app.post("/train_model/")
async def train_model():
    return {"message": "Modelo entrenado correctamente"}
