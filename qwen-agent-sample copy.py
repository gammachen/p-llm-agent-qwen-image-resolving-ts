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
        content_text = ''
        images = []
        
        # 处理消息内容
        for message in messages:
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', '')
            elif hasattr(message, 'role') and hasattr(message, 'content'):
                role = message.role
                content = message.content
            else:
                continue
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, ContentItem):
                        if item.text:
                            content_text += item.text + '\n'
                        if item.image:
                            # 检查是否为base64编码
                            if item.image.startswith('data:image/'):
                                # 对于base64编码的图像，我们保留它
                                images.append(item.image)
                            else:
                                # 对于文件路径，我们直接使用
                                images.append(item.image)
            else:
                content_text += str(content) + '\n'
        
        # 构建ollama消息
        ollama_message = {
            'role': 'user',
            'content': content_text.strip(),
        }
        
        # 如果有图像，添加到消息中
        if images:
            ollama_message['images'] = images
        
        ollama_messages.append(ollama_message)
    
        ollama_messages=[
            {
                'role': 'user',
                'content': "提取图片中的所有方程式等内容",
                'images': ["/private/var/folders/9l/kbk_mdlj0x5bcvscm_41pmbm0000gn/T/gradio/d88ef2f6ddad0bb40c20677406569983cbae029d8ee318fc06e16dcbc5c54514/test1.png"]
            }
        ]
        
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
        # 确保yield的是列表形式的Message对象
        if not isinstance(response, list):
            response = [response]
        yield response
        
    def _chat_with_functions(self, messages, functions, **kwargs):
        # 实现带函数调用的聊天
        # 对于简单情况，我们可以先将函数信息添加到消息中
        functions_info = f"可用函数: {str(functions)}"
        
        # 创建新的消息列表，添加函数信息
        messages_with_functions = copy.deepcopy(messages)
        
        # 在最后一条用户消息中添加函数信息
        if messages_with_functions and len(messages_with_functions) > 0:
            last_msg = messages_with_functions[-1]
            if hasattr(last_msg, 'role'):
                role = last_msg.role
            else:
                role = last_msg.get('role', '')
                
            if role == 'user':
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                else:
                    content = last_msg.get('content', '')
                    
                if isinstance(content, str):
                    if hasattr(last_msg, 'content'):
                        last_msg.content += '\n\n' + functions_info
                    else:
                        last_msg['content'] += '\n\n' + functions_info
                elif isinstance(content, list):
                    # 如果是ContentItem列表，添加一个新的文本项
                    if content and isinstance(content[0], ContentItem):
                        content.append(ContentItem(text=functions_info))
        
        # 调用普通聊天方法
        return self._chat_no_stream(messages_with_functions, **kwargs)


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
        # 校验WebUI传入的参数，必须为list，且包含图片
        assert isinstance(messages[-1]['content'], list)
        assert any([item.image for item in messages[-1]
                   ['content']]), '这个智体应用需要输入图片'

        response = []
        # 第1个Agent，将图片内容识别成文本
        new_messages = copy.deepcopy(messages)
        input_text = new_messages[-1].content[0]['text']
        input_image = new_messages[-1].content[1]['image']
        
        print(f'输入文本: {input_text}')
        print(f'图像路径: {input_image}')

        # 将图像路径从file://格式转换为本地路径
        image_path = input_image.replace('file://', '')
        
        # 我们已经在文件顶部导入了这些模块
        # 确保图像路径正确，并转换为base64编码格式
        # try:
        #     # 检查图像文件是否存在
        #     if os.path.exists(image_path):
        #         # 将图像转换为base64编码
        #         base64_image = encode_image_as_base64(image_path)
        #         print(f'图像已成功转换为base64编码')
        #     else:
        #         print(f'警告：图像文件不存在: {image_path}')
        #         base64_image = image_path
        # except Exception as e:
        #     print(f'转换图像时出错: {str(e)}')
        #     base64_image = image_path
        
        base64_image = image_path
        
        # 创建包含文本和编码后图像的内容列表
        new_messages[-1].content = [
            ContentItem(text=input_text),
            ContentItem(image=base64_image)
        ]
        
        # 使用image_agent识别图片中的数学问题
        for rsp in self.image_agent.run(new_messages, lang=lang, **kwargs):
            print(f'图像识别结果: {rsp}')
            yield rsp
        
        # 第2个Agent，求解文本中的数学问题
        response = rsp
        new_messages.extend(rsp)
        new_messages.append(Message('user', '根据以上文本内容求解数学题'))
        
        for rsp in self.math_agent.run(new_messages, lang=lang, **kwargs):
            print(f'数学计算结果: {rsp}')
            yield response + rsp


def app_gui():
    bot = Visual_solve_equations(llm=llm_config)
    WebUI(bot).run(
        server_name="192.168.31.122",
        server_port=7861
    )


if __name__ == '__main__':
    app_gui()
