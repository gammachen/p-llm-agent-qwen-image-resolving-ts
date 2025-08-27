import copy
import os
import ollama
from typing import Dict, Iterator, List, Optional, Union
from qwen_agent import Agent
from qwen_agent.tools import BaseTool
from qwen_agent.agents import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message, ContentItem
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

# 配置本地ollama服务的模型名称
llm_config = {
    'model': 'granite3.2-vision',
    'temperature': 0.1,
    'max_tokens': 2048
}

llm_config_2 = {
    'model': 'qwen:latest',
    'temperature': 0.1,
    'max_tokens': 2048
}

# 自定义LLM类，包装ollama的调用
class OllamaLLM(BaseChatModel):
    def __init__(self, model_config):
        super().__init__()
        self.model_name = model_config['model']
        self.temperature = model_config.get('temperature', 0.1)
        self.max_tokens = model_config.get('max_tokens', 2048)
    
    def _chat(self, messages, **kwargs):
        return self._chat_no_stream(messages, **kwargs)
    
    def _chat_no_stream(self, messages, **kwargs):
        # 转换qwen-agent的消息格式为ollama的消息格式
        ollama_messages = []
        
        # 处理消息内容
        for message in messages:
            # 确保消息是Message对象格式
            if isinstance(message, dict):
                role = message.get('role', 'user')
                content = message.get('content', '')
            elif isinstance(message, Message):
                role = message.role
                content = message.content
            else:
                # 如果是字符串或其他类型，创建一个新的Message对象
                role = 'user'
                content = str(message)
            
            # 处理内容
            content_text = ''
            images = []
            
            if isinstance(content, list):
                # 处理ContentItem列表
                for item in content:
                    if isinstance(item, ContentItem):
                        if item.text:
                            content_text += item.text + '\n'
                        if item.image:
                            # 检查是否为base64编码
                            if isinstance(item.image, str) and item.image.startswith('data:image/'):
                                images.append(item.image)
                            elif isinstance(item.image, str):
                                # 文件路径，直接使用
                                images.append(item.image)
                    elif isinstance(item, dict):
                        # 处理字典格式的内容
                        if 'text' in item:
                            content_text += item['text'] + '\n'
                        if 'image' in item:
                            images.append(item['image'])
            elif isinstance(content, str):
                content_text = content
            else:
                content_text = str(content)
            
            # 构建ollama消息
            ollama_message = {
                'role': role,
                'content': content_text.strip(),
            }
            
            # 如果有图像，添加到消息中
            if images:
                ollama_message['images'] = images
            
            ollama_messages.append(ollama_message)
        
        try:
            # 调用本地ollama服务
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                options={
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
            )
            
            # 返回响应，确保返回Message对象
            result_content = response['message']['content']
            print("模型响应:", result_content)
            
            return Message(role='assistant', content=result_content)
        except Exception as e:
            print(f"调用ollama服务时出错: {str(e)}")
            error_message = f"调用模型时出错: {str(e)}"
            return Message(role='assistant', content=error_message)
            
    def _chat_stream(self, messages, **kwargs):
        # 简单实现流式响应，返回非流式响应的生成器
        response = self._chat_no_stream(messages, **kwargs)
        # 确保返回的是Message对象
        if isinstance(response, Message):
            yield [response]
        elif isinstance(response, list):
            # 确保列表中的每个元素都是Message对象
            valid_messages = []
            for msg in response:
                if isinstance(msg, Message):
                    valid_messages.append(msg)
                elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    valid_messages.append(Message(role=msg['role'], content=msg['content']))
                else:
                    valid_messages.append(Message(role='assistant', content=str(msg)))
            yield valid_messages
        else:
            # 如果返回的是其他类型，包装成Message对象
            yield [Message(role='assistant', content=str(response))]
        
    def _chat_with_functions(self, messages, functions, **kwargs):
        # 实现带函数调用的聊天
        # 对于简单情况，我们可以先将函数信息添加到消息中
        functions_info = f"可用函数: {str(functions)}"
        
        # 创建新的消息列表，添加函数信息
        messages_with_functions = copy.deepcopy(messages)
        
        # 确保所有消息都是Message对象
        valid_messages = []
        for msg in messages_with_functions:
            if isinstance(msg, Message):
                valid_messages.append(msg)
            elif isinstance(msg, dict):
                valid_messages.append(Message(role=msg.get('role', 'user'), content=msg.get('content', '')))
            else:
                valid_messages.append(Message(role='user', content=str(msg)))
        
        # 在最后一条用户消息中添加函数信息
        if valid_messages and len(valid_messages) > 0:
            last_msg = valid_messages[-1]
            if last_msg.role == 'user':
                content = last_msg.content
                if isinstance(content, str):
                    last_msg.content = content + '\n\n' + functions_info
                elif isinstance(content, list):
                    # 如果是ContentItem列表，添加一个新的文本项
                    content.append(ContentItem(text=functions_info))
                else:
                    last_msg.content = str(content) + '\n\n' + functions_info
        
        # 调用普通聊天方法
        return self._chat_no_stream(valid_messages, **kwargs)


class Visual_solve_equations(Agent):
    def __init__(self,
                 function_list: Optional[
                     List[Union[str,
                                Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None):
        # 如果传入的是字典配置，使用自定义的OllamaLLM
        if isinstance(llm, dict):
            llm = OllamaLLM(llm)
        super().__init__(llm=llm)
        
        # 定义图片识别Agent
        self.image_agent = Assistant(llm=self.llm)
        # 定义数学计算Agent
        self.math_agent = Assistant(
            llm=self.llm,
            system_message='你扮演一个学生，' +
            '参考你学过的数学知识进行计算')

    def _run(self, messages: List[Message],
             lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:
        # 确保消息格式正确
        if not messages:
            yield [Message(role='assistant', content='没有收到消息')]
            return
            
        # 获取最后一条消息
        last_message = messages[-1]
        
        # 确保最后一条消息有正确的格式
        if not isinstance(last_message, Message):
            yield [Message(role='assistant', content='消息格式不正确')]
            return
            
        # 检查消息内容格式
        content = last_message.content
        if not isinstance(content, list):
            yield [Message(role='assistant', content='消息内容格式不正确')]
            return
            
        # 检查是否包含图片
        has_image = False
        for item in content:
            if isinstance(item, ContentItem) and item.image:
                has_image = True
                break
            elif isinstance(item, dict) and 'image' in item:
                has_image = True
                break
                
        if not has_image:
            yield [Message(role='assistant', content='这个智体应用需要输入图片')]
            return

        response = []
        # 第1个Agent，将图片内容识别成文本
        new_messages = copy.deepcopy(messages)
        
        # 安全地获取输入文本和图像
        input_text = "请识别图片中的数学方程式并转换为文本格式"
        input_image = ""
        
        # 从消息内容中提取文本和图像
        for item in content:
            if isinstance(item, ContentItem):
                if item.text and item.text.strip():
                    input_text = item.text
                if item.image:
                    input_image = item.image
            elif isinstance(item, dict):
                if 'text' in item and item['text'] and item['text'].strip():
                    input_text = item['text']
                if 'image' in item:
                    input_image = item['image']
        
        print(f'输入文本: {input_text}')
        print(f'图像路径: {input_image}')

        # 将图像路径从file://格式转换为本地路径
        image_path = str(input_image).replace('file://', '')
        
        # 确保图像路径正确，并转换为base64编码格式
        try:
            # 检查图像文件是否存在
            if os.path.exists(image_path):
                # 将图像转换为base64编码
                base64_image = encode_image_as_base64(image_path)
                print(f'图像已成功转换为base64编码')
            else:
                print(f'警告：图像文件不存在: {image_path}')
                base64_image = image_path
        except Exception as e:
            print(f'转换图像时出错: {str(e)}')
            base64_image = image_path
        
        # 创建包含文本和编码后图像的新消息
        image_message = Message(
            role='user',
            content=[
                ContentItem(text=input_text),
                ContentItem(image=base64_image)
            ]
        )
        
        # 使用image_agent识别图片中的数学问题
        image_messages = new_messages[:-1] + [image_message]
        
        for rsp in self.image_agent.run(image_messages, lang=lang, **kwargs):
            print(f'图像识别结果: {rsp}')
            yield rsp
        
        # 第2个Agent，求解文本中的数学问题
        if rsp and len(rsp) > 0:
            # 获取识别结果
            recognition_result = rsp[-1].content if isinstance(rsp[-1].content, str) else str(rsp[-1].content)
            
            # 创建新的消息用于数学计算
            math_message = Message(
                role='user',
                content=f"请根据以下内容求解数学问题：{recognition_result}"
            )
            
            math_messages = new_messages[:-1] + [math_message]
            
            for math_rsp in self.math_agent.run(math_messages, lang=lang, **kwargs):
                print(f'数学计算结果: {math_rsp}')
                yield math_rsp


def app_gui():
    bot = Visual_solve_equations(llm=llm_config)
    WebUI(bot).run(
        server_name="192.168.31.122",
        server_port=7861
    )


if __name__ == '__main__':
    app_gui()
