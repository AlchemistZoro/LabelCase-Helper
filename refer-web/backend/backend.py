import json
import re

import uvicorn
from fastapi import FastAPI, WebSocket

from refer import refer

app = FastAPI()


@app.websocket("/")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_text()

    data = json.loads(data)
    await ws.send_text(json.dumps(refer(data["lines"])))
    await ws.close()


if __name__ == "__main__":
    uvicorn.run(app='backend:app', host="127.0.0.1", port=8010, reload=True, debug=True)
