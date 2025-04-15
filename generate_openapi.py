# generate_openapi.py
from api.main import app
from fastapi.openapi.utils import get_openapi
import json

openapi_schema = get_openapi(
    title=app.title,
    version=app.version,
    routes=app.routes,
)

with open("docs/openapi.json", "w") as f:
    json.dump(openapi_schema, f)

print("✅ OpenAPI schema saved to docs/openapi.json")
