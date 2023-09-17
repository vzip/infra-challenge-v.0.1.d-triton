import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import models_config

class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self, model):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = model

    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)[0]

class ModelLoader:
    def __init__(self, models_config):
        self.models_config = models_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_save_models(self):
        for config in self.models_config:
            model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])
            model.to(self.device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

            sentence = 'Who are you voting for in 2020?'
            labels = config["model_labels"]

            inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                                return_tensors='pt', max_length=256,
                                                truncation=True, padding='max_length')

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            pt_model = PyTorch_to_TorchScript(model).eval()

            remove_attributes = []
            for key, value in vars(pt_model).items():
                if value is None:
                    remove_attributes.append(key)

            for key in remove_attributes:
                delattr(pt_model, key)

            traced_script_module = torch.jit.trace(pt_model, (input_ids, attention_mask), strict=False)

            model_path = os.path.join("/models", config["worker_name"], "1", "model.pt")
            traced_script_module.save(model_path)

if __name__ == "__main__":
    model_loader = ModelLoader(models_config)
    model_loader.load_and_save_models()
