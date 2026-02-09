from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules.chat import router as chat_router
from modules.admin import router as admin_router
from modules.auth import router as auth_router

app = FastAPI(
    title="FortiLM API",
    description="Security Middleware for Large Language Models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])

@app.get("/")
async def root():
    return {"message": "FortiLM API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "FortiLM API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

