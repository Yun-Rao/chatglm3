# backend/rag.py

import json
import torch
from datetime import datetime
from threading import Thread
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

MERGED_MODEL_PATH = "/root/autodl-tmp/ChatGLM/output/chatglm3-6b-merged"
EMBEDDING_MODEL   = "BAAI/bge-large-zh-v1.5"
VECTOR_DB_PATH    = "/root/autodl-tmp/ChatGLM/vector_db"
LONG_TERM_DB_PATH = "/root/autodl-tmp/ChatGLM/vector_db_longterm"

SCORE_THRESHOLD    = 0.5
LT_SCORE_THRESHOLD = 0.45
TOP_K              = 3
SHORT_TERM_TURNS   = 5
COT_TEMPERATURE    = 0.1
ANS_TEMPERATURE    = 0.3
COT_MAX_TOKENS     = 150
ANS_MAX_TOKENS     = 512

# 判断是否为实质性内容（值得存入记忆）
MIN_MEMORY_LEN = 30

_embeddings = _vectorstore = _lt_vectorstore = _tokenizer = _model = None


def initialize():
    global _embeddings, _vectorstore, _lt_vectorstore, _tokenizer, _model
    print("加载 Embedding 模型...")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("加载农业知识库...")
    _vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=_embeddings)
    print("加载长期记忆库...")
    _lt_vectorstore = Chroma(
        persist_directory=LONG_TERM_DB_PATH,
        collection_name="long_term_memory",
        embedding_function=_embeddings
    )
    print(f"加载语言模型：{MERGED_MODEL_PATH}")
    _tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MERGED_MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    _model.eval()
    print("模型初始化完成！")


# ==================== 记忆模块 ====================

AGRI_KEYWORDS = [
    "种植","施肥","农药","病虫害","防治","作物","水稻","小麦","玉米","蔬菜","果树","大棚",
    "养殖","猪","牛","羊","鸡","鸭","鱼","虾","水产","畜禽","饲料","疫病","疫苗",
    "土壤","灌溉","除草","收割","播种","育苗","嫁接","剪枝","授粉","农业","农田",
    "肥料","温室","有机","病害","虫害","草害","菌","霉","枯萎","腐烂",
    "芹菜","黄瓜","番茄","辣椒","棉花","大豆","花生","甘蔗","茶叶",
    "苹果","梨","桃","葡萄","草莓","牛奶","产蛋","孵化","繁殖","配种","驱虫","消毒"
]

def is_agri_question(question: str) -> bool:
    return any(kw in question for kw in AGRI_KEYWORDS)


def save_to_long_term_memory(user_id, question, answer, is_summary=False):
    """只有农业相关问题才存入长期记忆"""
    if len(answer) < MIN_MEMORY_LEN:
        return
    if not is_summary and not is_agri_question(question):
        return  # 闲聊不存记忆
    content = f"{'[摘要]' if is_summary else '问题'}：{question}\n内容：{answer[:400]}"
    metadata = {
        "user_id": str(user_id), "question": question,
        "timestamp": datetime.utcnow().isoformat(), "is_summary": str(is_summary)
    }
    _lt_vectorstore.add_texts(texts=[content], metadatas=[metadata])
    _lt_vectorstore.persist()


def retrieve_long_term_memory(user_id, question, top_k=2):
    try:
        results = _lt_vectorstore.similarity_search_with_score(
            question, k=top_k, filter={"user_id": str(user_id)})
        return [(doc, score) for doc, score in results if score <= LT_SCORE_THRESHOLD]
    except Exception:
        return []


def summarize_old_history(old_messages):
    if not old_messages:
        return ""
    dialog_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:200]}"
        for m in old_messages
    )
    prompt = f"将以下农业问答对话压缩为100字以内的摘要，只保留关键问题和结论：\n{dialog_text}"
    resp, _ = _model.chat(_tokenizer, prompt, history=[], max_new_tokens=150, temperature=0.1)
    return resp.strip()


def process_history_with_summary(user_id, session_history):
    total_turns = len(session_history) // 2
    if total_turns <= SHORT_TERM_TURNS:
        return _format_history(session_history), ""
    recent_count    = SHORT_TERM_TURNS * 2
    old_messages    = session_history[:-recent_count]
    recent_messages = session_history[-recent_count:]
    short_text = _format_history(recent_messages)
    summary = summarize_old_history(old_messages)
    if summary:
        topic = old_messages[0]["content"][:50] if old_messages else "对话历史"
        save_to_long_term_memory(user_id, topic, summary, is_summary=True)
    return short_text, summary


def _format_history(messages):
    if not messages:
        return ""
    return "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:200]}"
        for m in messages
    )


def retrieve_knowledge(question, top_k=TOP_K):
    results = _vectorstore.similarity_search_with_score(question, k=top_k)
    return [(doc, score) for doc, score in results if score <= SCORE_THRESHOLD]


def build_context(knowledge_docs, short_term, old_summary, long_term_docs):
    parts = []
    if old_summary:
        parts.append(f"[历史摘要]\n{old_summary}")
    if short_term:
        parts.append(f"[近期对话]\n{short_term}")
    if long_term_docs:
        lt_text = "\n".join(f"- {doc.page_content[:150]}" for doc, _ in long_term_docs)
        parts.append(f"[相关历史]\n{lt_text}")
    if knowledge_docs:
        kb_text = "\n".join(
            f"[{i+1}] {doc.page_content}" for i, (doc, _) in enumerate(knowledge_docs)
        )
        parts.append(f"[知识库]\n{kb_text}")
    return "\n\n".join(parts)


# ==================== 流式问答 ====================

def stream_query(question, user_id, session_history, temperature=ANS_TEMPERATURE, top_k=TOP_K):
    short_term_text, old_summary = process_history_with_summary(user_id, session_history)
    long_term_docs = retrieve_long_term_memory(user_id, question)
    knowledge_docs = retrieve_knowledge(question, top_k)
    used_turns     = min(len(session_history) // 2, SHORT_TERM_TURNS)
    context        = build_context(knowledge_docs, short_term_text, old_summary, long_term_docs)

    # ---- 第一次调用：思维链 ——只让模型列出关键因素，不给完整答案 ----
    if context:
        cot_prompt = (
            f"{context}\n\n"
            f"问题：{question}\n\n"
            f"请列出回答这个问题需要考虑的2-3个关键因素（每条不超过20字，不要给出完整答案）："
        )
    else:
        cot_prompt = (
            f"问题：{question}\n\n"
            f"请列出回答这个问题需要考虑的2-3个关键因素（每条不超过20字，不要给出完整答案）："
        )

    cot_text, _ = _model.chat(
        _tokenizer, cot_prompt,
        history=[],
        max_new_tokens=COT_MAX_TOKENS,
        temperature=COT_TEMPERATURE,
        top_p=0.9,
    )
    cot_text = cot_text.strip()
    yield json.dumps({"type": "thinking", "content": cot_text}, ensure_ascii=False) + "\n"

    # ---- 第二次调用：流式生成最终答案 ----
    if context:
        ans_prompt = (
            f"{context}\n\n"
            f"关键因素：{cot_text}\n\n"
            f"请针对问题「{question}」给出详细的专业回答："
        )
    else:
        ans_prompt = (
            f"关键因素：{cot_text}\n\n"
            f"请针对问题「{question}」给出详细的专业回答："
        )

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    try:
        inputs = _tokenizer.build_chat_input(ans_prompt, history=[], role="user")
        input_ids = inputs["input_ids"].to(_model.device)
    except Exception:
        input_ids = _tokenizer([ans_prompt], return_tensors="pt").input_ids.to(_model.device)

    thread = Thread(target=_model.generate, kwargs=dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=ANS_MAX_TOKENS,
        temperature=ANS_TEMPERATURE,  # 固定，不受前端 temperature 参数影响
        top_p=0.9,
        do_sample=True,
    ))
    thread.start()

    answer_buffer = []
    buf = ""
    for token in streamer:
        buf += token
        if len(buf) >= 8:
            answer_buffer.append(buf)
            yield json.dumps({"type": "answer", "content": buf}, ensure_ascii=False) + "\n"
            buf = ""

    if buf.strip():
        answer_buffer.append(buf)
        yield json.dumps({"type": "answer", "content": buf}, ensure_ascii=False) + "\n"

    thread.join()
    full_answer = "".join(answer_buffer).strip()
    if not full_answer:
        full_answer = "抱歉，暂时无法回答这个问题，请尝试换一种提问方式。"
        yield json.dumps({"type": "answer", "content": full_answer}, ensure_ascii=False) + "\n"

    # 只有农业问题才存入长期记忆
    save_to_long_term_memory(user_id, question, full_answer)

    sources = [
        {"score": round(float(s), 4),
         "instruction": doc.metadata.get("instruction", ""),
         "content": doc.page_content[:300]}
        for doc, s in knowledge_docs
    ]
    yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False) + "\n"
    yield json.dumps({
        "type": "meta",
        "long_term_used": len(long_term_docs) > 0,
        "short_term_turns": used_turns,
        "had_summary": bool(old_summary),
        "cot_analysis": cot_text,
    }, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}) + "\n"


def query(question, user_id, session_history, temperature=ANS_TEMPERATURE, top_k=TOP_K):
    thinking_parts, answer_parts, sources, meta = [], [], [], {}
    for chunk in stream_query(question, user_id, session_history, temperature, top_k):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            data = json.loads(chunk)
            t = data.get("type")
            if t == "thinking": thinking_parts.append(data["content"])
            elif t == "answer":  answer_parts.append(data["content"])
            elif t == "sources": sources = data["data"]
            elif t == "meta":    meta = data
        except Exception:
            pass
    return {
        "thinking": "".join(thinking_parts),
        "answer": "".join(answer_parts),
        "sources": sources,
        "long_term_used": meta.get("long_term_used", False),
        "short_term_turns": meta.get("short_term_turns", 0),
        "had_summary": meta.get("had_summary", False),
    }