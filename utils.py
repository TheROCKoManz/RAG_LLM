import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import subprocess

HOME = os.getcwd()

def install_requirements():
  os.environ['OPENAI_API_KEY']=input('Enter your <OPENAI-API-KEY>: ')
  try:
      subprocess.check_call(['pip', 'install', '-r', f'{HOME}/requirements.txt'])
      print("Requirements installation successful.")
  except subprocess.CalledProcessError as e:
      print(f"Error: Failed to install requirements. {e}")
  except FileNotFoundError:
      print("Error: 'pip' command not found. Please make sure Python and pip are correctly installed.")


def load_data():
  documents = f'/{HOME}/RAG_LLM/KnowledgeBase'
  loader = DirectoryLoader(documents)
  data = loader.load()
  return data
  

def extract_embedding_vectorstore(data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)

  all_splits = text_splitter.split_documents(data)
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

  return vectorstore

def build_model():
  data = load_data()
  vectorstore = extract_embedding_vectorstore(data)
  template = """Use the following pieces of context to answer the question at the end. 
              You are a Retrieval Augmented Generative AI who is trained to answer questions based on a Document regarding PAN card informations.
              You have been created by Lord TheROCKoManz.
              If you don't know the answer, just say that you don't know, don't try to make up an answer. 
              Use three sentences maximum and keep the answer as concise as possible. 
              Always say "\nThanks for asking!" at the end of the answer. 
              {context}
              Question: {question}
              Helpful Answer:"""

  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

  llm = OpenAI(model_name="text-davinci-002", callbacks=[StreamingStdOutCallbackHandler()], temperature = 0.7)

  qa_model = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

  return qa_model


