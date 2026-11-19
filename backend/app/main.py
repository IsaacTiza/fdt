import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.ml_service import load_model
from app.db.mongo import ping_database
from app.routers import predict, analytics

app = FastAPI(title="Fraud Detection API")

# Local dev + the deployed Vercel frontend. Extra origins can be added on Render
# via the CORS_ORIGINS env var (comma-separated), e.g. a custom domain.
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "https://fdt-two.vercel.app",
]
EXTRA_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
ALLOWED_ORIGINS = DEFAULT_ORIGINS + [o for o in EXTRA_ORIGINS if o not in DEFAULT_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predict.router)
app.include_router(analytics.router)


@app.on_event("startup")
async def startup():
    # No model file -> load_model() returns None and /predict answers 503.
    # A model that exists but mismatches the schema raises here, on purpose.
    load_model()
    connected = await ping_database()
    if not connected:
        print("[WARNING] MongoDB connection failed at startup.")


@app.get("/health")
async def health():
    return {"status": "ok"}