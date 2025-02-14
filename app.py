import streamlit as st
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from huggingface_hub import snapshot_download

# 페이지 설정
st.set_page_config(page_title="매뉴얼 Q&A", layout="wide")
st.title("매뉴얼 Q&A 시스템")

# 시크릿 키 체크
if 'TOGETHER_API_KEY' not in st.secrets or 'HUGGINGFACE_TOKEN' not in st.secrets:
    st.error('스트림릿 Secrets에 TOGETHER_API_KEY와 HUGGINGFACE_TOKEN이 설정되어 있지 않습니다.')
    st.stop()

# 시크릿에서 API 키 가져오기
together_api_key = st.secrets["TOGETHER_API_KEY"]
huggingface_token = st.secrets["HUGGINGFACE_TOKEN"]

# 세션 스테이트 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_models():
    # TogetherLLM 설정
    llm = TogetherLLM(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=together_api_key,
        temperature=0,
        max_tokens=2048
    )
    
    # 임베딩 모델 설정
    embed_model = HuggingFaceEmbedding( model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", token=huggingface_token )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

@st.cache_resource
def load_index():
    # 허깅페이스에서 인덱스 다운로드
    repo_id = "blockenters/manual-index-paraphrase"
    local_dir = "manual-index-paraphrase"

    # 허깅페이스에 있는 데이터를 로컬에 다운로드한다.
    snapshot_download(
        repo_id= repo_id,
        local_dir= local_dir,
        repo_type= 'dataset',
        token= huggingface_token
    )

    # 다운로드한 폴더를 메모리에 올린다. 
    storage_context = StorageContext.from_defaults(persist_dir= local_dir)
    index = load_index_from_storage(storage_context)    

    return index

# 모델 초기화
llm, embed_model = initialize_models()

# 로딩 스피너 표시
with st.spinner('인덱스를 로딩중입니다...'):
    index = load_index()
    query_engine = index.as_query_engine()

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력
if prompt := st.chat_input("질문을 입력하세요."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답
    with st.chat_message("assistant"):
        with st.spinner('답변을 생성중입니다...'):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

# 사이드바에 사용 설명 추가
with st.sidebar:
    st.header("사용 방법")
    st.write("1. 하단의 채팅창에 질문을 입력하세요")
    st.write("2. Enter 키를 누르면 답변이 생성됩니다")
    st.write("3. 답변이 생성되는 동안 잠시 기다려주세요")
    
    # 채팅 초기화 버튼
    if st.button("채팅 초기화"):
        st.session_state.messages = []
        st.rerun() 