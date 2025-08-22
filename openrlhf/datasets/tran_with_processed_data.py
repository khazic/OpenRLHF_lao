import os
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader

class ProcessedSFTDataset(Dataset):
    def __init__(self, processed_data_dir: str):
        self.data = []
        self.load_processed_data(processed_data_dir)
    
    def load_processed_data(self, data_dir: str):
        """Load all processed batch data"""
        print("Loading preprocessed data...")
        
        # Load metadata
        with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Load all batches
        for batch_idx in range(metadata["total_batches"]):
            batch_file = f"batch_{batch_idx:06d}.{metadata['save_format']}"
            batch_path = os.path.join(data_dir, batch_file)
            
            if metadata['save_format'] == 'pickle':
                with open(batch_path, 'rb') as f:
                    batch_data = pickle.load(f)
            else:
                with open(batch_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
            
            self.data.extend(batch_data)
            print(f"Loaded batch {batch_idx + 1}/{metadata['total_batches']}")
        
        print(f"Total loaded samples: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["input_ids"]),
            torch.tensor(item["attention_mask"]),
            torch.tensor(item["loss_mask"])
        )

# Usage example
processed_dataset = ProcessedSFTDataset("./processed_sft_data")
train_dataloader = DataLoader(
    processed_dataset, 
    batch_size=8, 
    shuffle=True,
    collate_fn=lambda x: (
        torch.stack([item[0] for item in x]),
        torch.stack([item[1] for item in x]),
        torch.stack([item[2] for item in x])
    )
)