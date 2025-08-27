import os
import copy
import ollama
from typing import Dict, Iterator, List, Optional, Union
from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message, ContentItem
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

# 配置本地ollama服务的模型
VISION_MODEL_CONFIG = {
    'model': 'granite3.2-vision',
    'temperature': 0.1,
    'max_tokens': 2048
}

MATH_MODEL_CONFIG = {
    'model': 'qwen2:latest',
    'temperature': 0.1,
    'max_tokens': 2048
}

class OllamaVisionLLM(BaseChatModel):
    """用于图像识别的LLM包装器"""
    
    def __init__(self, model_config: Dict):
        super().__init__()
        self.model_name = model_config['model']
        self.temperature = model_config.get('temperature', 0.1)
        self.max_tokens = model_config.get('max_tokens', 2048)
    
    def _chat(self, messages: List[Message], **kwargs) -> List[Message]:
        """非流式聊天接口"""
        return self._chat_no_stream(messages, **kwargs)
    
    def _chat_no_stream(self, messages: List[Message], **kwargs) -> List[Message]:
        """非流式聊天实现"""
        return self._process_messages(messages)
    
    def _chat_stream(self, messages: List[Message], **kwargs) -> Iterator[List[Message]]:
        """流式聊天接口"""
        result = self._process_messages(messages)
        yield result
    
    def _chat_with_functions(self, messages: List[Message], functions, **kwargs) -> List[Message]:
        """带函数调用的聊天接口"""
        # 简化实现，直接调用普通聊天
        return self._chat_no_stream(messages, **kwargs)
    
    def _process_messages(self, messages: List[Message]) -> List[Message]:
        """处理消息并调用ollama"""
        try:
            # 提取最后一条消息的内容
            if not messages:
                return [Message(role='assistant', content='没有收到消息')]
            
            last_message = messages[-1]
            content = last_message.content
            
            # 处理多模态内容
            text_content = ""
            image_data = None
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, ContentItem):
                        if item.text:
                            text_content = item.text
                        if item.image:
                            image_data = item.image
                    elif isinstance(item, dict):
                        if 'text' in item:
                            text_content = item['text']
                        if 'image' in item:
                            image_data = item['image']
            elif isinstance(content, str):
                text_content = content
            
            # 构建ollama消息
            ollama_messages = []
            ollama_msg = {
                'role': 'user',
                'content': text_content or "请识别图片中的数学内容"
            }
            
            # 添加图像数据
            if image_data:
                if image_data.startswith('data:image/'):
                    # base64格式
                    ollama_msg['images'] = [image_data]
                elif os.path.exists(image_data):
                    # 文件路径，转换为base64
                    base64_image = encode_image_as_base64(image_data)
                    ollama_msg['images'] = [base64_image]
                else:
                    # 直接作为图像数据
                    ollama_msg['images'] = [str(image_data)]
            
            ollama_messages.append(ollama_msg)
            
            # 调用ollama
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                options={
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
            )
            
            result = response['message']['content']
            return [Message(role='assistant', content=result)]
            
        except Exception as e:
            error_msg = f"调用ollama服务出错: {str(e)}"
            return [Message(role='assistant', content=error_msg)]

class MathSolverAgent(Agent):
    """数学解题智能体"""
    
    def __init__(self):
        super().__init__()
        self.name = "数学解题智能体"
        # 创建图像识别agent
        self.vision_agent = Assistant(
            llm=OllamaVisionLLM(VISION_MODEL_CONFIG),
            system_message="你是一个图像识别专家，专门识别数学方程式、公式和题目。请将图片中的数学内容准确转换为文本格式。"
        )
        
        # 创建数学解题agent
        self.math_agent = Assistant(
            llm=OllamaVisionLLM(MATH_MODEL_CONFIG),
            system_message="你是一个数学解题专家，专门解决方程组、代数、几何等数学问题。请详细解答给出的数学题目，包括解题步骤和最终答案。"
        )
    
    def _run(self, messages: List[Message], **kwargs) -> Iterator[List[Message]]:
        """主运行逻辑"""
        if not messages:
            yield [Message(role='assistant', content='没有收到消息')]
            return
        
        # 获取最后一条消息
        last_message = messages[-1]
        
        # 检查是否包含图片
        has_image = False
        content = last_message.content
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, ContentItem) and item.image:
                    has_image = True
                    break
                elif isinstance(item, dict) and 'image' in item:
                    has_image = True
                    break
        
        if not has_image:
            yield [Message(role='assistant', content='请上传包含数学题目的图片')]
            return
        
        # 步骤1: 使用vision agent识别图片内容
        print("步骤1: 开始识别图片中的数学内容...")
        
        # 创建识别请求
        vision_messages = copy.deepcopy(messages)
        vision_result = None
        
        for response in self.vision_agent.run(vision_messages):
            vision_result = response
            yield response
        
        if not vision_result:
            yield [Message(role='assistant', content='图片识别失败')]
            return
        
        # 获取识别结果
        recognition_text = vision_result[-1].content if vision_result else ""
        print(f"识别结果: {recognition_text}")
        
        # 步骤2: 使用math agent解题
        print("步骤2: 开始解答数学问题...")
        
        # 创建数学解题请求
        math_message = Message(
            role='user',
            content=f"请详细解答以下数学问题：\n{recognition_text}"
        )
        
        math_messages = messages[:-1] + [math_message]
        
        for math_response in self.math_agent.run(math_messages):
            yield math_response

def launch_app():
    """启动Web应用"""
    agent = MathSolverAgent()
    
    WebUI(agent).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )

if __name__ == "__main__":
    launch_app()