import numpy as np
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def create_signatures_numba(texts_encoded, shingle_size, num_perm, prime, a_params, b_params):
    n_texts = len(texts_encoded)
    signatures = np.full((n_texts, num_perm), prime, dtype=np.int32)
    
    for text_idx in prange(n_texts):
        text = texts_encoded[text_idx]
        text_len = len(text)
        
        if text_len < shingle_size:
            continue
        
        sig = np.full(num_perm, prime, dtype=np.int32)
        
        for i in range(text_len - shingle_size + 1):
            shingle_hash = 0
            for j in range(shingle_size):
                shingle_hash = shingle_hash * 31 + text[i + j]
            
            shingle_hash = shingle_hash % prime
            
            for perm_idx in range(num_perm):
                h = (a_params[perm_idx] * shingle_hash + b_params[perm_idx]) % prime
                if h < sig[perm_idx]:
                    sig[perm_idx] = h
        
        signatures[text_idx] = sig
    
    return signatures


class SignatureGenerator:
    def __init__(self, prime=2147483647):
        self.prime = prime
    
    def create_signatures(self, texts, num_perm, shingle_size):
        texts_encoded = []
        for text in texts:
            encoded = np.array([ord(c) for c in text[:500]], dtype=np.int32)
            texts_encoded.append(encoded)
        
        max_len = max(len(t) for t in texts_encoded)
        texts_padded = np.zeros((len(texts_encoded), max_len), dtype=np.int32)
        for i, t in enumerate(texts_encoded):
            texts_padded[i, :len(t)] = t
        
        np.random.seed(42)
        a_params = np.random.randint(1, self.prime, size=num_perm, dtype=np.int64)
        b_params = np.random.randint(0, self.prime, size=num_perm, dtype=np.int64)
        
        signatures = create_signatures_numba(
            texts_padded,
            shingle_size,
            num_perm,
            self.prime,
            a_params,
            b_params
        )
        
        return signatures