from transformers import TrainingArguments, Trainer

# helps in preparing the inputs, attention masks, and labels in the correct format.
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
from textSummarization.entity import ModelTrainerConfig
import os
import torch

torch.cuda.empty_cache()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # Legacy -> False done 
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = T5Tokenizer.from_pretrained(self.config.model_ckpt,legacy=False)
        model_t5 = T5ForConditionalGeneration.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

# Params.yaml file read
        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # ) 


        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=2 , fp16=True,  gradient_checkpointing=True
        ) 
        # Gradient accumulation reduced 16 -> 8 -> 2

        
        


        trainer = Trainer(model=model_t5, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_t5.save_pretrained(os.path.join(self.config.root_dir,"t5-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))

        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
