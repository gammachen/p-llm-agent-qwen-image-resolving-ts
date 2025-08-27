import os
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

# 创建图像识别agent
vision_agent = Assistant(
    llm={'model': 'granite3.2-vision'},
    system_message="你是一个图像识别专家，专门识别数学方程式、公式和题目。请将图片中的数学内容准确转换为文本格式。"
)

# 创建数学解题agent
math_agent = Assistant(
    llm={'model': 'qwen2:latest'},
    system_message="你是一个数学解题专家，专门解决方程组、代数、几何等数学问题。请详细解答给出的数学题目，包括解题步骤和最终答案。"
)

class MathSolverAgent:
    """数学解题智能体"""
    
    def __init__(self):
        self.name = "数学解题智能体"
    
    def run(self, messages):
        """运行数学解题流程"""
        if not messages:
            return ["请上传包含数学题目的图片"]
        
        # 获取最后一条消息
        last_msg = messages[-1]
        
        # 检查是否有图片
        if not hasattr(last_msg, 'content') or not last_msg.content:
            return ["请上传图片"]
        
        content = last_msg.content
        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'image') and item.image:
                    # 处理图片
                    image_path = str(item.image).replace('file://', '')
                    
                    try:
                        if os.path.exists(image_path):
                            base64_image = encode_image_as_base64(image_path)
                        else:
                            base64_image = image_path
                    except:
                        base64_image = image_path
                    
                    print("🔍 识别图片中的数学内容...")
                    
                    # 步骤1: 图像识别
                    vision_result = vision_agent.run([
                        {
                            'role': 'user',
                            'content': [
                                {'text': '请识别图片中的数学方程式或题目'},
                                {'image': base64_image}
                            ]
                        }
                    ])
                    
                    # 获取识别结果
                    recognition = ""
                    for response in vision_result:
                        recognition = response[-1]['content'] if response else ""
                    
                    print(f"✅ 识别结果: {recognition}")
                    
                    print("🧮 解答数学问题...")
                    
                    # 步骤2: 数学解题
                    math_result = math_agent.run([
                        {
                            'role': 'user',
                            'content': f"请详细解答以下数学问题：\n{recognition}"
                        }
                    ])
                    
                    return list(math_result)
        
        return ["请上传包含数学题目的图片"]

# 创建智能体实例
agent = MathSolverAgent()

if __name__ == "__main__":
    # 启动Web界面
    WebUI(agent.run).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )