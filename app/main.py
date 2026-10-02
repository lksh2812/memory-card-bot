"""FastAPI app. For now just a health check; routes are added in later steps."""

from fastapi import FastAPI

app = FastAPI(title="Memory Card Voice Bot")


@app.get("/health")
async def health():
    return {"status": "ok"}
