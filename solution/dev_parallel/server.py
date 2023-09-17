# server.py
import asyncio
import logging
from typing import Union, List, Dict
from fastapi import FastAPI
from worker import Worker
from responder import Responder
from main_handler import process_data
from config import config  # Import the configuration

app = FastAPI()

# Configure logging
logger = logging.getLogger('server')
logger.setLevel(logging.DEBUG)

futures_dict = {}
worker_queues = [asyncio.Queue() for _ in range(5)]  # Create 5 worker queues
result_queue = asyncio.Queue()  # Create a single result queue

workers = []

@app.on_event("startup")
async def startup_event():
    # Start the workers
    for i, queue in enumerate(worker_queues):
        logging.info(f'Starting worker with queue {id(queue)}.')
        worker_config = config["workers"][i]  # Get the configuration for this worker
        event = asyncio.Event()
        worker_instance = Worker(worker_config, queue, result_queue, event, logger)
        workers.append(worker_instance)

    for worker in workers:
        asyncio.create_task(worker.start())

    # Start the Responder
    responder_instance = Responder(futures_dict, result_queue, logger)
    asyncio.create_task(responder_instance.start())

    logging.info('Server started, running result listener')

@app.post("/process")
async def process_endpoint(data: Union[str, List[Dict[str, str]], Dict[str, str]]):
    return await process_data(data, worker_queues, futures_dict, logger)
