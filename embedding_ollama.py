# embedding_ollama.py
import requests
import traceback
from typing import List

class OllamaEmbeddings:
    """
    Ollama 本地服务提供的 Embedding 接口，
    最终拼出形如: http://localhost:11434/api/embeddings
    即 base_url + "/embeddings"
    但是按文档，好像/embeddings接口已经被废弃了，现在是/embed才对，实际测试都可以用，视情况而定
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
        url = f"{self.base_url}/embeddings"
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
            raise Exception(f"Ollama embeddings request error: {e}\n{traceback.format_exc()}")
