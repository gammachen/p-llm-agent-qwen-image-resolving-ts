import os
import ollama
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

# 创建图像识别agent
vision_agent = Assistant(
    llm='granite3.2-vision',
    system_message="你是一个图像识别专家，专门识别数学方程式、公式和题目。请将图片中的数学内容准确转换为文本格式。"
)

# 创建数学解题agent
math_agent = Assistant(
    llm='qwen2:latest',
    system_message="你是一个数学解题专家，专门解决方程组、代数、几何等数学问题。请详细解答给出的数学题目，包括解题步骤和最终答案。"
)

class MathSolverPipeline:
    """数学解题管道"""
    
    def __init__(self):
        self.name = "数学解题智能体"
    
    def run(self, messages):
        """运行数学解题流程"""
        if not messages:
            return ["请上传包含数学题目的图片"]
        
        # 获取最后一条消息
        last_msg = messages[-1]
        
        # 检查是否有图片
        image_data = None
        
        if hasattr(last_msg, 'content') and last_msg.content:
            content = last_msg.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'image') and item.image:
                        image_path = str(item.image).replace('file://', '')
                        
                        try:
                            if os.path.exists(image_path):
                                image_data = encode_image_as_base64(image_path)
                            else:
                                image_data = image_path
                        except:
                            image_data = image_path
                        break
        
        if not image_data:
            return ["请上传包含数学题目的图片"]
        
        print("🔍 步骤1: 识别图片中的数学内容...")
        
        # 步骤1: 图像识别
        vision_messages = [{
            'role': 'user',
            'content': [
                {'text': '请识别图片中的数学方程式或题目，并转换为清晰的文本格式'},
                {'image': image_data}
            ]
        }]
        
        # 运行图像识别
        vision_result = []
        for response in vision_agent.run(vision_messages):
            vision_result.extend(response)
        
        if not vision_result:
            return ["图片识别失败"]
        
        recognized_text = vision_result[-1]['content']
        print(f"✅ 识别结果: {recognized_text}")
        
        print("🧮 步骤2: 解答数学问题...")
        
        # 步骤2: 数学解题
        math_messages = [{
            'role': 'user',
            'content': f"请详细解答以下数学问题：\n{recognized_text}"
        }]
        
        # 运行数学解题
        math_result = []
        for response in math_agent.run(math_messages):
            math_result.extend(response)
        
        return math_result

# 创建智能体实例
solver = MathSolverPipeline()

if __name__ == "__main__":
    # 启动Web界面
    WebUI(solver.run).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )