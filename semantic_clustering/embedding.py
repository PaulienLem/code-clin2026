# import torch
# import numpy as np
# from transformers import AutoModel, AutoTokenizer
# from tqdm import tqdm
# import time
# import psutil

# class EmbeddingGenerator:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = None
#         self.tokenizer = None
        
#     def load_model(self):
#         start = time.time()
#         self.model = AutoModel.from_pretrained(
#             self.config.model_name, 
#             trust_remote_code=True
#         ).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.model_name, 
#             trust_remote_code=True
#         )
#         self.model.eval()
#         load_time = time.time() - start
        
#         return {
#             'model_load_time': load_time,
#             'device': str(self.device),
#             'model_params': sum(p.numel() for p in self.model.parameters())
#         }
    
#     def embed_texts(self, texts, verse_ids):
#         start = time.time()
        
#         clean_texts = []
#         clean_ids = []
#         for text, vid in zip(texts, verse_ids):
#             if isinstance(text, str) and len(text.strip()) > 0:
#                 clean_texts.append(text.strip())
#                 clean_ids.append(vid)
#             elif text is not None:
#                 clean_texts.append(str(text).strip())
#                 clean_ids.append(vid)
        
#         if len(clean_texts) == 0:
#             return {
#                 'embeddings': np.array([]),
#                 'id_mapping': [],
#                 'embed_time': 0.0,
#                 'embedding_dim': 0,
#                 'memory_gb': 0.0
#             }
        
#         embeddings = []
#         id_mapping = []
        
#         n_batches = (len(clean_texts) + self.config.batch_size - 1) // self.config.batch_size
        
#         with torch.no_grad():
#             for i in tqdm(range(0, len(clean_texts), self.config.batch_size), 
#                          desc='embedding', total=n_batches):
#                 batch_texts = clean_texts[i:i + self.config.batch_size]
#                 batch_ids = clean_ids[i:i + self.config.batch_size]
                
#                 inputs = self.tokenizer(
#                     batch_texts,
#                     padding=True,
#                     truncation=True,
#                     max_length=self.config.max_length,
#                     return_tensors='pt'
#                 ).to(self.device)
                
#                 outputs = self.model(**inputs)
#                 batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
#                 embeddings.append(batch_embeddings)
#                 id_mapping.extend(batch_ids)
        
#         embeddings = np.vstack(embeddings)
#         embed_time = time.time() - start
        
#         return {
#             'embeddings': embeddings,
#             'id_mapping': id_mapping,
#             'embed_time': embed_time,
#             'embedding_dim': embeddings.shape[1],
#             'memory_gb': psutil.Process().memory_info().rss / 1024**3
#         }
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
import psutil
from torch.cuda.amp import autocast

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.use_amp = torch.cuda.is_available()
        self.use_multi_gpu = torch.cuda.device_count() > 1
        
    def load_model(self):
        start = time.time()
        self.model = AutoModel.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        )
        
        if self.use_multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.eval()
        
        load_time = time.time() - start
        
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        return {
            'model_load_time': load_time,
            'device': str(self.device),
            'n_gpus': n_gpus,
            'use_amp': self.use_amp,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
    
    def _estimate_optimal_batch_size(self):
        if not torch.cuda.is_available():
            return self.config.batch_size
        
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 40:
                return min(256, self.config.batch_size * 8)
            elif gpu_memory >= 24:
                return min(128, self.config.batch_size * 4)
            elif gpu_memory >= 16:
                return min(64, self.config.batch_size * 2)
            elif gpu_memory >= 11:
                return min(48, self.config.batch_size * 1.5)
            else:
                return self.config.batch_size
        except:
            return self.config.batch_size
    
    def embed_texts(self, texts, verse_ids):
        start = time.time()
        
        clean_texts = []
        clean_ids = []
        for text, vid in zip(texts, verse_ids):
            if isinstance(text, str) and len(text.strip()) > 0:
                clean_texts.append(text.strip())
                clean_ids.append(vid)
            elif text is not None:
                clean_texts.append(str(text).strip())
                clean_ids.append(vid)
        
        if len(clean_texts) == 0:
            return {
                'embeddings': np.array([]),
                'id_mapping': [],
                'embed_time': 0.0,
                'embedding_dim': 0,
                'memory_gb': 0.0
            }
        
        optimal_batch_size = self._estimate_optimal_batch_size()
        
        embeddings = []
        id_mapping = []
        
        n_batches = (len(clean_texts) + optimal_batch_size - 1) // optimal_batch_size
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(clean_texts), optimal_batch_size), 
                         desc='embedding', total=n_batches):
                batch_texts = clean_texts[i:i + optimal_batch_size]
                batch_ids = clean_ids[i:i + optimal_batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(**inputs)
                        if isinstance(outputs, tuple):
                            batch_embeddings = outputs[0][:, 0, :].cpu().float().numpy()
                        else:
                            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
                else:
                    outputs = self.model(**inputs)
                    if isinstance(outputs, tuple):
                        batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
                    else:
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
                id_mapping.extend(batch_ids)
                
                if torch.cuda.is_available():
                    del inputs, outputs
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
        
        embeddings = np.vstack(embeddings)
        embed_time = time.time() - start
        
        return {
            'embeddings': embeddings,
            'id_mapping': id_mapping,
            'embed_time': embed_time,
            'embedding_dim': embeddings.shape[1],
            'batch_size_used': optimal_batch_size,
            'memory_gb': psutil.Process().memory_info().rss / 1024**3
        }