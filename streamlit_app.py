from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
# Specify the path to your PDF file
from langchain_community.document_loaders import PyPDFLoader
#from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_core.documents import Document
import streamlit as st
import os


def get_file_text(pdf_path):
  loader = PyPDFLoader(pdf_path)
  docs=loader.load()
  documents=[]
  documents.extend((Document(page_content=doc.page_content,metadata=doc.metadata) for doc in docs))
  return documents





def get_file_text(pdf_path):
  loader = PyPDFLoader(pdf_path)
  docs=loader.load()
  documents=[]
  documents.extend((Document(page_content=doc.page_content,metadata=doc.metadata) for doc in docs))
  return documents





# def get_file_text(file_path):
#   # Initialize the loader
#   #loader = DirectoryLoader(file_path)
#   loader = DirectoryLoader(file_path, glob="**/*.md")
#   documents_files = loader.load()
#   len(documents_files)
#   documents = []
#   # Load the document
  
#   documents.extend((Document(page_content=doc.content,metadata=doc.Metadata) for doc in documents_files))

  
#   return documents




def main():
    # Create a session state to keep track of whether the app is running
    
     #model = ChatOpenAI(model="gpt-3.5-turbo")
    MYPATH = "Kashmir Digital Policy 2024-2030 Sept 6.pdf"
    pdf_path = MYPATH
    documents = get_file_text(pdf_path)
    vectorstore2 = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
    )


    retriever = RunnableLambda(vectorstore2.similarity_search).bind(k=1)  # select top result

    retriever.batch(["what is total population of AJK", "What is size of male and female population"])
    #openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    openai.api_key =  os.getenv("OPENAI_API_KEY")
    st.write(openai.api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    api_key=api_key)


    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
 
  
    
    # Create a session state to keep track of whether the app is running
    if 'running' not in st.session_state:
        st.session_state.running = True

    st.title("AJK digital Policy 2024-2030")
    st.markdown(
            """
            <style>
            output {
                font-size: 1.3rem !important;
            }
            input {
                font-size: 1.2rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    # Show the input form if the app is running
    if st.session_state.running:
        # Take input from the user
        user_input = st.text_input("As any Question about AJK digital Policy:")
        submit = st.button('submit')
        # Button to stop the loop
        if st.button("Exit"):
            st.session_state.running = False
        # Display the input
        if user_input and submit:          
            response = rag_chain.invoke(user_input)
            st.write(response.content)

    # When the user clicks 'Exit', stop the loop
    else:
        st.write("You have exited the session.")
        if st.button("Restart"):
            st.session_state.running = True

if __name__ == "__main__":
    main()

