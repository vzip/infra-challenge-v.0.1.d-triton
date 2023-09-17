import logging
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import tritonclient.http as tritonhttpclient
import numpy as np


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class Worker:
    def __init__(self, worker_config, message_queue, results_queue, event, logger):
        self.logger = logger
        self.worker_name = worker_config["worker_name"]
        self.model_name = worker_config["model_name"]
        self.model_labels = worker_config["model_labels"]
        self.event = event

        self.message_queue = message_queue
        self.results_queue = results_queue

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.event.set()

    async def start(self):
        triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000")
        while True:
            if not self.message_queue.empty():
                message = await self.message_queue.get()
                await self.process_message(message, triton_client)
            else:
                await asyncio.sleep(0.01)

    async def process_message(self, message, triton_client):
        body = json.loads(message) if isinstance(message, str) else message
        text = body['data']['data']
        correlation_id = body['correlation_id']

        if not isinstance(text, list):
            text = [text]
        while len(text) < 4:
            text.append('')

        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=256, truncation=True)

        input_ids_data = inputs['input_ids'].numpy()
        attention_mask_data = inputs['attention_mask'].numpy()

        infer_input_ids = tritonhttpclient.InferInput('input__0', input_ids_data.shape, 'INT32')
        infer_input_ids.set_data_from_numpy(input_ids_data.astype(np.int32))

        infer_attention_mask = tritonhttpclient.InferInput('input__1', attention_mask_data.shape, 'INT32')
        infer_attention_mask.set_data_from_numpy(attention_mask_data.astype(np.int32))
        response = triton_client.infer(
            model_name=self.worker_name,
            inputs=[infer_input_ids, infer_attention_mask],
            outputs=[tritonhttpclient.InferRequestedOutput('output__0')],
            model_version="1"
        )

        output_data = response.as_numpy('output__0')

        print(output_data.shape)
        print(output_data)
        prediction = np.argmax(output_data, axis=-1)
        print(prediction)
        score = np.max(output_data, axis=-1)
        median_score = np.median(score)
        label = self.model_labels[np.argmax(prediction)]

        result_key = self.model_name.split('/')[0]
        result = {result_key: {"score": median_score, "label": label}}

        results_dict = {
            "correlation_id": correlation_id,
            "worker": self.worker_name,
            "result": result
        }
        results_dict = json.dumps(results_dict, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

        await self.results_queue.put(results_dict)
