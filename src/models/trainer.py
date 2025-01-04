from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset

class LlamaFineTuner:
    def __init__(self, model_name="meta-llama/Llama-2-7b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup(self):
        """Initialize the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def prepare_data(self, data):
        """Convert raw data into training format."""
        formatted_data = []
        for item in data:
            text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Output: {item['output']}"
            formatted_data.append({"text": text})
        return Dataset.from_list(formatted_data)
    
    def train(self, dataset, output_dir="./results"):
        """Fine-tune the model."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=100,
            save_total_limit=2,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=10,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)