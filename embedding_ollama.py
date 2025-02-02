# embedding_ollama.py
import requests
from typing import List

class OllamaEmbeddings:
    """
    Ollama 本地服务提供的 Embedding 接口，
    最终拼出形如: http://localhost:11434/api/embed
    即 base_url + "/embed"
    """

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量将多段文本转换为embedding向量
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single_document(text))
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        兼容langchain的接口写法
        """
        return self.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        将单条 query 转换为 embedding 向量
        """
        return self.embed_single_document(query)

    def embed_single_document(self, text: str) -> List[float]:
        """
        调用 Ollama 本地服务接口，获取文本的 embedding。
        """
        url = f"{self.base_url}/embed"
        data = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            if "embedding" not in result:
                raise ValueError("No 'embedding' field in Ollama response.")
            return result["embedding"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama embeddings request error: {e}")
