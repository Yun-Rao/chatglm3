# backend/main.py
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
import rag
from database import engine, get_db, SessionLocal
from auth import (
    hash_password, verify_password, create_token,
    get_current_user, get_admin_user
)

models.Base.metadata.create_all(bind=engine)
rag.initialize()

app = FastAPI(title="农业知识 RAG 问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/root/autodl-tmp/ChatGLM/rag_system/frontend"), name="static")


# ==================== Pydantic 模型 ====================

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "新对话"

class SessionRenameRequest(BaseModel):
    title: str

class QueryRequest(BaseModel):
    session_id: int
    question: str
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 3


# ==================== 前端页面 ====================

@app.get("/")
def index():
    return FileResponse("/root/autodl-tmp/ChatGLM/rag_system/frontend/index.html")


# ==================== 认证接口 ====================

@app.post("/api/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == req.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    user = models.User(
        username=req.username, email=req.email,
        password=hash_password(req.password), is_admin=False
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token(user.id, user.username, user.is_admin)
    return {"token": token, "username": user.username, "is_admin": user.is_admin}


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == req.username).first()
    if not user or not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")
    token = create_token(user.id, user.username, user.is_admin)
    return {"token": token, "username": user.username, "is_admin": user.is_admin}


@app.get("/api/auth/me")
def me(current_user: models.User = Depends(get_current_user)):
    return {
        "id": current_user.id, "username": current_user.username,
        "email": current_user.email, "is_admin": current_user.is_admin,
        "created_at": current_user.created_at
    }


# ==================== 会话接口 ====================

@app.get("/api/sessions")
def list_sessions(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(models.Session).filter(
        models.Session.user_id == current_user.id
    ).order_by(models.Session.updated_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at,
             "updated_at": s.updated_at, "message_count": len(s.messages)} for s in sessions]


@app.post("/api/sessions")
def create_session(req: SessionCreateRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = models.Session(user_id=current_user.id, title=req.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"id": session.id, "title": session.title, "created_at": session.created_at}


@app.put("/api/sessions/{session_id}")
def rename_session(session_id: int, req: SessionRenameRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(models.Session).filter(models.Session.id == session_id, models.Session.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    session.title = req.title
    db.commit()
    return {"message": "重命名成功"}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(models.Session).filter(models.Session.id == session_id, models.Session.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(session)
    db.commit()
    return {"message": "删除成功"}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(models.Session).filter(models.Session.id == session_id, models.Session.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = db.query(models.Message).filter(models.Message.session_id == session_id).order_by(models.Message.created_at).all()
    return [
        {
            "id": m.id, "role": m.role, "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "cot": m.cot or "",
            "created_at": m.created_at
        }
        for m in messages
    ]


# ==================== 流式问答接口 ====================

@app.post("/api/query/stream")
async def query_stream(
    req: QueryRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(models.Session).filter(
        models.Session.id == req.session_id,
        models.Session.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 保存用户消息
    user_msg = models.Message(session_id=req.session_id, role="user", content=req.question)
    db.add(user_msg)
    if session.title == "新对话":
        session.title = req.question[:30] + ("..." if len(req.question) > 30 else "")
    session.updated_at = datetime.utcnow()
    db.commit()

    # 获取会话历史（去掉刚保存的最新这条）
    all_messages = db.query(models.Message).filter(
        models.Message.session_id == req.session_id
    ).order_by(models.Message.created_at).all()
    session_history = [{"role": m.role, "content": m.content} for m in all_messages[:-1]]

    user_id = current_user.id
    session_id = req.session_id

    def event_generator():
        full_answer = []
        full_sources = []
        full_thinking = []

        for raw in rag.stream_query(
            question=req.question,
            user_id=user_id,
            session_history=session_history,
            temperature=req.temperature,
            top_k=req.top_k
        ):
            if isinstance(raw, str):
                try:
                    event = json.loads(raw.strip())
                except Exception:
                    continue
            else:
                event = raw

            evt_type = event.get("type", "")

            if evt_type == "thinking":
                # 收集思维链，不推送到前端（等 meta 一起发出）
                full_thinking.append(event.get("content", ""))

            elif evt_type in ("answer", "token"):
                full_answer.append(event.get("content", ""))
                yield f"data: {json.dumps({'type': 'token', 'content': event.get('content', '')}, ensure_ascii=False)}\n\n"

            elif evt_type == "sources":
                full_sources = event.get("content") or event.get("data", [])
                yield f"data: {json.dumps({'type': 'sources', 'content': full_sources}, ensure_ascii=False)}\n\n"

            elif evt_type == "meta":
                # 把完整思维链附在 meta 里一次性发给前端
                cot_text = "".join(full_thinking) or event.get("cot_analysis", "")
                meta_content = {
                    "long_term_used": event.get("long_term_used", False),
                    "short_term_turns": event.get("short_term_turns", 0),
                    "has_summary": event.get("had_summary", False),
                    "cot_analysis": cot_text,
                }
                yield f"data: {json.dumps({'type': 'meta', 'content': meta_content}, ensure_ascii=False)}\n\n"

            elif evt_type == "done":
                answer_text = "".join(full_answer)
                cot_text = "".join(full_thinking)
                save_db = SessionLocal()
                try:
                    assistant_msg = models.Message(
                        session_id=session_id,
                        role="assistant",
                        content=answer_text,
                        sources=json.dumps(full_sources, ensure_ascii=False),
                        cot=cot_text  # 正确存入思维链
                    )
                    save_db.add(assistant_msg)
                    save_db.commit()
                finally:
                    save_db.close()
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ==================== 管理员接口 ====================

@app.get("/api/admin/users")
def admin_list_users(admin: models.User = Depends(get_admin_user), db: Session = Depends(get_db)):
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return [{"id": u.id, "username": u.username, "email": u.email, "is_admin": u.is_admin,
             "is_active": u.is_active, "created_at": u.created_at, "session_count": len(u.sessions)} for u in users]


@app.put("/api/admin/users/{user_id}/toggle")
def admin_toggle_user(user_id: int, admin: models.User = Depends(get_admin_user), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="不能禁用自己")
    user.is_active = not user.is_active
    db.commit()
    return {"message": "操作成功", "is_active": user.is_active}


@app.put("/api/admin/users/{user_id}/set_admin")
def admin_set_admin(user_id: int, admin: models.User = Depends(get_admin_user), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.is_admin = not user.is_admin
    db.commit()
    return {"message": "操作成功", "is_admin": user.is_admin}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)