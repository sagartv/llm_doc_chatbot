## LLM Document Expert

A Streamlit App that uses Langchain, OpenAI Embeddings, GPT, and Pinecone Vector Databases to process a user-provided document.
The document is chunked, and then converted to word embeddings using OpenAI Embeddings. The embeddings are inserted into a Pinecone Index which is deleted after runtime.
Langchain is used to retrieve information through the QA

Upload the document in the sidebar: .pdf, .docx, and .txt files are supported.
You can also control chunk size to improve the quality of the responses.
