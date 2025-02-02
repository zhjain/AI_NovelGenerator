import logging
import requests
from typing import List
import traceback

class OllamaEmbeddings:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url  # 这里应形如 http://localhost:11434/api (不再含 /v1)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_single_document(text))
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多段文本转换为向量列表
        """
        embeddings = []
        for text in texts:
            emb = self.embed_single_document(text)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        将单条 query 转换为 embedding 向量
        """
        return self.embed_single_document(query)

    def embed_single_document(self, text: str) -> List[float]:
        """
        调用 Ollama 本地服务接口，获取文本的 embedding。
        这里统一改为请求:  [base_url]/embed
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
            print(result)

            # 检查返回结果是否包含 'embedding' 字段
            if "embedding" not in result:
                logging.warning(f"No 'embedding' field in response. Returning empty embedding.")
                return []  # 返回空列表
            return result["embedding"]

        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama embeddings request error: {e}")
            logging.error(f"Request URL: {url}")
            logging.error(f"Request Data: {data}")
            logging.error("Full error details:\n" + traceback.format_exc())
            return []

        except ValueError as e:
            logging.error(f"Invalid response structure: {e}")
            logging.error(f"Response content: {response.text}")
            logging.error("Full error details:\n" + traceback.format_exc())
            return []

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            logging.error("Full error details:\n" + traceback.format_exc())
            return []
