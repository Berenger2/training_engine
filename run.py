from src.app import app
import uvicorn
from multiprocessing import Process
from src.mlf import start_mlflow_ui

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=5030)

if __name__ == "__main__":
    p1 = Process(target=start_fastapi)
    p2 = Process(target=start_mlflow_ui)

    p1.start()
    p2.start()

    p1.join()
    p2.join()