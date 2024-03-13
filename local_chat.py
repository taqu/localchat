import io
import logging
import os
import asyncio
import openpyxl
from pptx import Presentation

import streamlit as st
import transformers
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import BaseSingleActionAgent
from langchain.agents import LLMSingleActionAgent
from langchain.agents import Tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import create_react_agent
from langchain.agents import initialize_agent
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.prompt import PROMPT
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import AIMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.tools import StructuredTool
from langchain.tools import Tool
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.llms.base import OpenAI
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_base = os.environ['API_BASE']
vectordb_host = os.environ['VECTORDB_HOST']
vectordb_port = os.environ['VECTORDB_PORT']
global_cache_dir = 'models'
global_log_verbosity = True
logger = logging.getLogger(st.__name__)
logger.setLevel(logging.INFO)
model_id_gemma = 0
model_id_gpt3 = 1
language_ja = 0
language_en = 1

@st.cache_resource
def get_local_embeddings():
    """
    Get a local embedding model.

    Returns
    -------
    embeddings : model instance
    """
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(
        model_name = 'intfloat/multilingual-e5-large',
        model_kwargs = model_kwargs,
        cache_folder=global_cache_dir)

@st.cache_resource
def get_vectordb_web():
    """
    Get persistent vector db for web scrapping.

    Returns
    -------
    persistent vector db for web search : db instance
    """
    model_kwargs = {'device': 'cpu'}
    embeddings = get_local_embeddings()
    return Chroma(embedding_function=embeddings, persist_directory='chroma_web')

@st.cache_resource
def get_vectordb_doc():
    """
    Get persistent vector db for documents
    Returns
    -------
    persistent vector db for documents : db instance
    """
    model_kwargs = {'device': 'cpu'}
    embeddings = get_local_embeddings()
    return Chroma(embedding_function=embeddings, persist_directory='chroma_doc')

@st.cache_resource
def get_vectordb_server(collection_name):
    """
    Get a client server model db.
    Returns
    -------
    persistent vector : db instance
    """
    embeddings = get_local_embeddings()
    client_settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host=vectordb_host,
        chroma_server_http_port=vectordb_port
    )
    return Chroma(
        collection_name=collection_name,
        client_settings=client_settings,
        embedding_function=embeddings
    )

@st.cache_resource
def get_google_api():
    return GoogleSearchAPIWrapper()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.history

#@st.cache_resource
#def get_trans_ja_en():
#    model = MarianMTModel.from_pretrained(pretrained_model_name_or_path='staka/fugumt-ja-en', cache_dir=global_cache_dir)
#    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='staka/fugumt-ja-en', cache_dir=global_cache_dir)
#    return pipeline('translation', model=model, tokenizer=tokenizer)

#@st.cache_resource
#def get_trans_en_ja():
#    model = MarianMTModel.from_pretrained(pretrained_model_name_or_path='staka/fugumt-en-ja', cache_dir=global_cache_dir)
#    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='staka/fugumt-en-ja', cache_dir=global_cache_dir)
#    return pipeline('translation', model=model, tokenizer=tokenizer)

#@st.cache_resource
#def get_fasttext():
#    return fasttext.load_model('models/lid.176.bin')

def get_stop_sequences():
    """
    Get stop sequences for a model.
    Returns
    -------
    stop sequences
    """
    if model_id_gemma == st.session_state.model_id:
        return ["### 入力","\n\n### 指示"]
    
    if model_id_gpt3 == st.session_state.model_id:
        return None

### Prompt templates
####################
CHAT_TEMPLATE_GEMMA_JA = "あなたは誠実で優秀な日本人のAIです。ユーザのメッセージに対して、短く豊かな返信文を作成してください。"
CHAT_PROMPT_GEMMA_JA = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            CHAT_TEMPLATE_GEMMA_JA
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

CHAT_TEMPLATE_GEMMA_EN = "You are a honest and excelent AI. Please compose short and rich replies to human messages."
CHAT_PROMPT_GEMMA_EN = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            CHAT_TEMPLATE_GEMMA_EN
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

def get_prompt_template_conversation(language:int):
    if model_id_gemma == st.session_state.model_id:
        if language_ja == language:
            return CHAT_PROMPT_GEMMA_JA
        else:
            return CHAT_PROMPT_GEMMA_EN

    if model_id_gpt3 == st.session_state.model_id:
        return PROMPT

DEFAULT_SEARCH_TEMPLATE_JA =\
"""<s>[INST] <<SYS>>
あなたはGoogle検索の結果を改善するアシスタントです。
<</SYS>>


次の質問に似たGoogle検索のクエリを3つ作ってください。出力は番号付きのリストで返してください。: {question} [/INST]"""

DEFAULT_SEARCH_PROMPT_JA = PromptTemplate(
    input_variables=["question"],
    template=DEFAULT_SEARCH_TEMPLATE_JA,
)

DEFAULT_SEARCH_TEMPLATE_EN =\
"""<s>[INST] <<SYS>>
You are an assistant tasked with improving Google search results.
<</SYS>>


Generate THREE Google search queries that are similar to this question. The output should be a numbered list of questions.: {question} [/INST]"""

DEFAULT_SEARCH_PROMPT_EN = PromptTemplate(
    input_variables=["question"],
    template=DEFAULT_SEARCH_TEMPLATE_EN,
)

def get_prompt_template_websearch():
    if model_id_gpt3 == st.session_state.model_id:
        return None
    if model_id_gemma == st.session_state.model_id:
        return None

    if language_ja == st.session_state.language:
        return DEFAULT_SEARCH_PROMPT_JA
    else:
        return DEFAULT_SEARCH_PROMPT_EN

#def predict_language(text:str):
#    model = get_fasttext()
#    text = text.replace('\n', '')
#    text = text.replace('\r', '')
#    label, prob = model.predict(text)
#    if len(label)<=0 or not label[0]:
#        return 'en'
#    return label[0].replace('__label__', '')

#def trans_ja_en(text):
#    tlock = threading.Lock()
#    with tlock:
#        trans_ja_en = get_trans_ja_en()
#        translated = trans_ja_en(text) 
#        return translated[0]['translation_text']

#def trans_en_ja(text):
#    tlock = threading.Lock()
#    with tlock:
#        trans_en_ja = get_trans_en_ja()
#        translated = trans_en_ja(text)
#        return translated[0]['translation_text']

def split_text(text:str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    return splitter.split_text(text)

def load_file(file:io.BytesIO):
    content = None
    if 'application/pdf' == file.type:
        pdf_reader = PdfReader(file)
        content = '\n'.join([page.extract_text() for page in pdf_reader.pages])
    elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' == file.type:
        book = openpyxl.load_workbook(file)
        sheets = book.sheetnames
        content = []
        for sheet_name in sheets:
            sheet = book[sheet_name]
            cell_list = [f"{cell.value}" for cells in tuple(sheet.columns) for cell in cells]
            content.extend(cell_list)
    elif 'application/vnd.openxmlformats-officedocument.presentationml.presentation' == file.type:
        presentation = Presentation(file)
        content = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                for par in shape.text_frame.paragraphs:
                    for run in par.runs:
                        content.append(run.text)
    elif 'text/plain' == file.type or 'text/csv' == file.type:
        content = file.read().decode('UTF-8')
    else:
        return None
    file.seek(0)
    return content

def embeddig_file(file:io.BytesIO):
    text = load_file(file)
    chunks = split_text(text)
    vector_db = get_vectordb_doc()
    vector_db.add_texts(chunks)
    results = vector_db.similarity_search("Please summarize comparizon of SMAA and other AA methods.")
    print(results)
    print('embeddig_file end')

def clear_session_state():
    if 'memory' in st.session_state:
        st.session_state.memory.clear()
    else:
        st.session_state.memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", return_messages=True)
    if 'history' in st.session_state:
        st.session_state.history.clear()
    else:
        st.session_state.history = ChatMessageHistory()
    st.session_state.model = model_id_gemma
    st.session_state.mode = 0
    st.session_state.language = language_ja 
    st.session_state.upload_file = None

def init_page():
    st.set_page_config(
            page_title='Chat Agent',
            page_icon='😊',
            layout='wide',
    )
    st.header('😊 Chat Agent 😊')
    st.sidebar.title("Options")

def init_options():
    ### Choose a model
    container = st.sidebar.container(border=True)
    model = container.radio('モデル:', ('gemma-7b-it', 'GPT-3.5'))
    temperature = container.slider('Temerature:', min_value=0.0, max_value=1.0, value=0.1)

    if model == 'gemma-7b-it':
        model_name = 'eramax/gemma-7b-it:q4_k_m'
        st.session_state.model_id = model_id_gemma
        st.session_state.model = ChatOllama(
            model=model_name,
            base_url=api_base,
            temperature=temperature,
            verbose=global_log_verbosity)

    else:
        model_name = 'gpt-3.5-turbo'
        st.session_state.model_id = model_id_gpt3
        st.session_state.model = OpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=2048,
            verbose=global_log_verbosity)

    ### Choose a language
    container = st.sidebar.container(border=True)
    modes = {0: "日本語", 1: "English"}
    def language_format_func(option):
        return modes[option]

    st.session_state.language = container.selectbox("Language:", options=list(modes.keys()), format_func=language_format_func)

    ### Choose a mode
    container = st.sidebar.container(border=True)
    modes = {0: "Chat", 1: "検索", 2: "ドキュメント"}
    def format_func(option):
        return modes[option]

    st.session_state.mode = container.selectbox("Mode:", options=list(modes.keys()), format_func=format_func)

    ### Mode specific options
    if 0 == st.session_state.mode:
        pass
    elif 1 == st.session_state.mode:
        st.session_state.use_googlesearch = container.checkbox('Google検索')
    elif 2 == st.session_state.mode:
        st.session_state.upload_file = container.file_uploader(label='ドキュメント:', type=['pdf', 'txt', 'text', 'xlsx', 'csv', 'html', 'ppt', 'pptx'])
        if st.session_state.upload_file:
            embeddig_file(st.session_state.upload_file)
    else:
        pass

def init_messages():
    clear_button = st.sidebar.button('Clear Conversation', key='clear')
    if clear_button or 'history' not in st.session_state:
        clear_session_state()

def invoke_chat_chain(user_input:str):
    prompt = get_prompt_template_conversation(st.session_state.language)
    chain = prompt | st.session_state.model | StrOutputParser()
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='history',
        )
    return with_message_history.invoke({'input': user_input}, {'configurable': {'session_id': '[0]'}})

def invoke_search_chain(user_input:str):
    vectorstore = get_vectordb_web()
    search = get_google_api()
    llm = st.session_state.model
    web_research_retriever = WebResearchRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        search=search,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40),
        num_search_results=3,
        prompt=get_prompt_template_websearch(),
    )
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=web_research_retriever,
        max_tokens_limit = 1000,
        verbose=global_log_verbosity,
        chain_type = 'stuff',
        chain_type_kwargs={'verbose': global_log_verbosity},
    )
    result = qa_chain.invoke(input=user_input)
    return result['answer']

def invoke_document_chain(user_input:str):
    vectorstore = get_vectordb_doc()
    llm = st.session_state.model
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        max_tokens_limit = 1000,
        return_source_documents = True,
        verbose=global_log_verbosity,
        chain_type_kwargs={'verbose': global_log_verbosity},
        )
    result = qa_chain.invoke(input=user_input)
    print(result)
    return 'response'

async def output_messages():
    """
    Output session messages.
    """
    messages = await st.session_state.history.aget_messages()
    #messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:
            pass

def do_chat_mode():
    if user_input := st.chat_input('Ask me anything!'):
        with st.spinner('Chat Agent is typing ...'):
            response = invoke_chat_chain(user_input=user_input)

    asyncio.run(output_messages())

def do_search_mode():
    if user_input := st.chat_input('Ask me anything!'):
        with st.spinner('Chat Agent is typing ...'):
            response = invoke_search_chain(user_input=user_input)
        st.session_state.history.add_user_message(HumanMessage(content=user_input))
        st.session_state.history.add_ai_message(AIMessage(content=response))

    asyncio.run(output_messages())

def do_document_mode():
    if st.session_state.upload_file:
        if user_input := st.chat_input('Ask me about the file'):
            with st.spinner('Chat Agent is typing ...'):
                response = invoke_document_chain(user_input=user_input)

            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=response))
    else:
        st.write('First, upload a file')

    asyncio.run(output_messages())

def main():
    init_page()
    init_options()
    init_messages()
    if 0 == st.session_state.mode:
        do_chat_mode()
    elif 1 == st.session_state.mode:
        do_search_mode()
    elif 2 == st.session_state.mode:
        do_document_mode()

if __name__ == '__main__':
    main()