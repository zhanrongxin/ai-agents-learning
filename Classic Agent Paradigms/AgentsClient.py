import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    LLM客户端
    """
    def __init__(self, model: str = None, apikey: str = None, baseUrl: str = None, timeout: int = None) -> None:
        """
        初始化LLM客户端
        """
        self.model = model or os.getenv("DASHSCOPE_MODEL")
        self.apikey = apikey or os.getenv("DASHSCOPE_API_KEY")
        self.baseUrl = baseUrl or os.getenv("DASHSCOPE_BASE_URL")
        self.timeout = timeout or int(os.getenv("DASHSCOPE_TIMEOUT", 60))
        # 校验必要参数
        if not all([self.model, self.apikey, self.baseUrl]):
            raise ValueError("模型、API密钥和基础URL不能为空")

        self.client = OpenAI(
            api_key=self.apikey,
            base_url=self.baseUrl,
            timeout=self.timeout,
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用LLM模型
        """
        print(f"正在调用 LLM 模型:  {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            # 处理流式响应
            print("LLM 模型调用成功，开始接收流式响应...")
            collection_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collection_content.append(content)
            print("\nLLM 流式响应结束\n")
            return "".join(collection_content)
        
        except Exception as e:
            print(f"调用模型{self.model}时出错: {e}")
            return ""