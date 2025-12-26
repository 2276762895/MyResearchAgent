import streamlit as st
import os
# --- 【新增】设置国内镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# --- 页面设置 ---
st.set_page_config(page_title="科研论文智能助手", layout="wide")
st.title("🎓 课题组科研助手 - 文献阅读版")

# --- 侧边栏：设置与文件上传 ---
with st.sidebar:
    st.header("⚙️ 设置")
    # 输入 DeepSeek API Key
    api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    st.markdown("[点击申请 DeepSeek API](https://platform.deepseek.com/)")

    st.divider()

    st.header("📂 上传文献")
    uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"])


# --- 核心逻辑函数 ---

@st.cache_resource
def process_pdf(file, api_key):
    if not file or not api_key:
        return None

    print("1. [开始] 正在保存临时文件...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    print(f"2. [加载] 正在读取PDF: {tmp_path} ...")
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    print(f"   -> PDF读取完成，共 {len(docs)} 页")

    print("3. [切分] 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"   -> 切分完成，共生成 {len(splits)} 个文本块")

    print("4. [模型] 正在加载 Embedding 模型 (这一步最容易卡)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("   -> 模型加载成功！")
    except Exception as e:
        print(f"   -> ❌ 模型加载失败: {e}")
        raise e

    print("5. [存储] 正在写入向量数据库 (FAISS)...")
    try:
        # 使用 FAISS 替代 Chroma，不需要 SQLite 支持，不会闪退
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        print("   -> 数据库写入成功！(已切换为 FAISS)")
    except Exception as e:
        print(f"   -> ❌ FAISS 写入失败: {e}")
        raise e

    print("6. [连接] 正在初始化 DeepSeek...")
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.3
    )

    print("7. [完成] 准备就绪！")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    os.remove(tmp_path)
    return qa_chain

# --- 主界面逻辑 ---

if uploaded_file and api_key:
    with st.spinner("正在阅读论文，请稍候... (第一次加载模型可能需要1分钟)"):
        # 处理PDF
        qa_chain = process_pdf(uploaded_file, api_key)

    st.success("✅ 论文已读取，快来问我问题吧！")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 获取用户输入
    if prompt := st.chat_input("这篇论文的主要贡献是什么？"):
        # 1. 显示用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 调用模型回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                source_docs = response["source_documents"]

                # 拼接引用来源（可选）
                source_text = "\n\n> **参考片段：**\n"
                for i, doc in enumerate(source_docs):
                    source_text += f"> {i + 1}. Page {doc.metadata['page']}: {doc.page_content[:100]}...\n"

                full_response = answer + source_text
                st.markdown(full_response)

        # 3. 保存助手回答
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("👈 请在左侧输入API Key并上传PDF文件开始。")