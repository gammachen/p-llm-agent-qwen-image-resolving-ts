import os
import copy
import ollama
from typing import List, Iterator
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

class MathSolverSystem:
    """数学解题系统"""
    
    def __init__(self):
        # 图像识别agent
        self.vision_agent = Assistant(
            llm={'model': 'granite3.2-vision'},
            system_message="你是一个图像识别专家，专门识别数学方程式、公式和题目。请将图片中的数学内容准确转换为文本格式。"
        )
        
        # 数学解题agent
        self.math_agent = Assistant(
            llm={'model': 'qwen2:latest'},
            system_message="你是一个数学解题专家，专门解决方程组、代数、几何等数学问题。请详细解答给出的数学题目，包括解题步骤和最终答案。"
        )
    
    def solve_math_from_image(self, messages: List[Message]) -> Iterator[List[Message]]:
        """从图片解题的完整流程"""
        if not messages:
            yield [Message(role='assistant', content='请上传图片')]
            return
        
        # 获取最后一条消息
        last_msg = messages[-1]
        
        # 检查是否有图片
        has_image = False
        image_content = None
        
        content = last_msg.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, ContentItem) and item.image:
                    has_image = True
                    image_content = item.image
                    break
        
        if not has_image:
            yield [Message(role='assistant', content='请上传包含数学题目的图片')]
            return
        
        print("步骤1: 识别图片中的数学内容...")
        
        # 处理图片
        image_path = str(image_content).replace('file://', '')
        
        try:
            if os.path.exists(image_path):
                base64_image = encode_image_as_base64(image_path)
            else:
                base64_image = image_path
        except Exception as e:
            base64_image = image_path
        
        # 步骤1: 图像识别
        vision_msg = Message(
            role='user',
            content=[
                ContentItem(text="请识别图片中的数学方程式或题目，并转换为清晰的文本格式"),
                ContentItem(image=base64_image)
            ]
        )
        
        recognition_result = None
        for response in self.vision_agent.run([vision_msg]):
            recognition_result = response
            yield response
        
        if not recognition_result:
            yield [Message(role='assistant', content='图片识别失败')]
            return
        
        # 获取识别文本
        recognized_text = recognition_result[-1].content
        print(f"识别结果: {recognized_text}")
        
        # 步骤2: 数学解题
        print("步骤2: 解答数学问题...")
        
        math_msg = Message(
            role='user',
            content=f"请详细解答以下数学问题：\n{recognized_text}"
        )
        
        for math_response in self.math_agent.run([math_msg]):
            yield math_response

def main():
    """主函数"""
    system = MathSolverSystem()
    
    # 创建主agent
    main_agent = Assistant(
        llm={'model': 'qwen2:latest'},
        function_list=[system.solve_math_from_image]
    )
    
    # 启动Web界面
    WebUI(system.solve_math_from_image).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )

if __name__ == "__main__":
    main()