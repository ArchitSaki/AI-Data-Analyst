
from fastapi import FastAPI

from app.api.upload import router as upload_router
from app.api.chat import router as chat_router


app = FastAPI(
    title="AI Data Analyst API",
    description="An AI-powered platform for exploring and analyzing CSV datasets.",
    version="0.1.0",
)

app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to AI Data Analyst",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
