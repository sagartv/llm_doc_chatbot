import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

#Loads File with .pdf, .docx, and .txt extensions
def load_file(file):
    import os
    name, extension = os.path.splitext(file)
    print("extension is ",extension)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading{file}')
        loader = Dox2txtLoader(file)
        
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
        
    else:
        print('Format Not Supported.')
        return None
    data = loader.load()
    return data

def delete_pinecone_index(index_name = "docchatapp1551v2"):
    import pinecone
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    if index_name in pc.list_indexes().names():
        print(f'Deleting index {index_name}...',end = '')
        pc.delete_index(index_name)
    else:
        print(f'Index {index_name} not found')




def fetch_and_store_embeddings(chunks, index_name = "docchatapp1551v2"):
    import pinecone
    from pinecone import Pinecone, PodSpec
    from langchain.vectorstores import Pinecone as Pineconevs
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    pc = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings...', end = '')
        vector_store = Pineconevs.from_existing_index(index_name, embeddings)
        print('ok')
        return vector_store
    else:
        print(f'Creating Index {index_name} and embeddings...', end = '')
        pc.create_index(index_name, dimension = 1536, metric = 'cosine', spec=PodSpec(
		environment='gcp-starter'
	))
        vector_store = Pineconevs.from_documents(chunks, embeddings, index_name = index_name)
        print('OK')
        return vector_store


#Function to chunk document
def chunk_data(data, chunk_size = 256, chunk_overlap = 10):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks =text_splitter.split_documents(data)
    return chunks

#Queries the Vector Store using RetrievalQA
def ask_and_get_answer(vector_store, q, k = 3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature = 0.1)
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwards = {'k' : k})
    chain = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)
    answer = chain.run(q)
    return answer

#Calculate the cost of embedding the text
def get_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    return (total_tokens, total_tokens/1000 * 0.0004)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        delete_pinecone_index()

    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override = True)
    st.set_page_config(
    page_title="LLM Document Expert",
    initial_sidebar_state="expanded",
)

    st.subheader('LLM Document Expert')
    st.write('Upload a pdf, txt, or docx using the sidebar to the left to ask the LLM expert questions. This can be a Resume, a novel, or almost anything under the sun.')
    st.divider()
    st.write('You will also need an OpenAI API Key and a Pinecone API Key')
    st.write('This app retrieves the contents of your file, chunks them according to the parameters you specify, and then inserts them into a Vector Database using Pinecone.')
    st.write('Answers are retrieved using Langchain.')
    st.divider()

    with st.sidebar:
        openai_api_key = st.text_input('OpenAI API Key:', type = 'password')
        pinecone_api_key = st.text_input('Pinecone API Key:', type = 'password')

        if openai_api_key and pinecone_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
        
        uploaded_file = st.file_uploader('Upload a file: ', type = ['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value = 100, max_value = 2048, value = 512, on_change = clear_history)
        k = st.number_input('k', min_value = 1, max_value = 20, value = 3, on_change = clear_history)
        add_data = st.button('Upload File', on_click = clear_history)

        if uploaded_file and add_data:
            with st.spinner('Processing File...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('files/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_file(file_name)
                if data:
                    print("file is ",data)
                    chunks = chunk_data(data = data, chunk_size = chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                    tokens, embedding_cost = get_embedding_cost(texts = chunks)
                    st.write(f'Embedding cost: ${embedding_cost:.4f}')
                    vector_store = fetch_and_store_embeddings(index_name = "docchatapp1551v2", chunks = chunks)
                    st.session_state.vs = vector_store
                    st.success('Document processed, chunked, and vectorized successfully')
                else:
                    st.write('Error: Document not processed')
    query = st.text_input('Ask the Expert a question from the content of your file:')
    if query:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store,query,k)
            st.text_area('LLM Expert: ',answer)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {query} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        h = st.session_state.history
        st.button('Delete History and Pinecone Index', on_click = clear_history)
        if not st.session_state['history']=='':
            with st.expander("History:"):
                st.text_area(label = "Chat History", value = h, key = 'history', height = 500)
                
        














