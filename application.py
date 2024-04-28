from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

print(f'\n\nHi! There is RAG for csv based on chromadb as vectordb and llama2 as LLM.\nPlease, <cd> to workfolder for creating local vactordb inside. If you choose your directory then input your CSV file path here...\n!!!CAUTION!!!\t If the csv file is too large, it will take a long time to embed all the data.')
CSV_PATH = input()



loader = CSVLoader(CSV_PATH)
big_df = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 100,
    length_function = len,
    add_start_index = True
)

chunks = text_splitter.split_documents(big_df)


embedings = OllamaEmbeddings(
    model='all-minilm'
)


from exp import create_folder_for_db
name = 'vectordb'
create_folder_for_db(name)

chromadb = Chroma.from_documents(chunks, embedings, persist_directory = name, collection_name = name)


message_history = ChatMessageHistory()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama

chain = ConversationalRetrievalChain.from_llm(
    ChatOllama(model="llama2"),
    chain_type="stuff",
    retriever=chromadb.as_retriever(),
    memory=memory,
    return_source_documents=True,
)

loop = True
while loop == True:
    query = input('Ask your question here...\t(<exit> to escape)\n')
    if query == 'exit':
        break
    else:
        res = chain.invoke(query)
        print(res['answer'])
print('Goodbye!')
