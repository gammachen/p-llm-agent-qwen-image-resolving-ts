import os
import ollama
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

# 创建数学解题智能体
class MathSolver:
    def __init__(self):
        self.name = "数学解题智能体"
    
    def solve(self, messages):
        """解题流程"""
        if not messages:
            return ["请上传图片"]
        
        # 获取最后一条消息
        last_msg = messages[-1]
        
        # 检查是否有图片
        if hasattr(last_msg, 'content') and last_msg.content:
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
                        
                        # 步骤1: 图像识别
                        print("🔍 识别图片中的数学内容...")
                        
                        # 调用granite3.2-vision进行图像识别
                        recognition_result = ollama.chat(
                            model='granite3.2-vision',
                            messages=[{
                                'role': 'user',
                                'content': '请识别图片中的数学方程式或题目，并转换为清晰的文本格式',
                                'images': [base64_image]
                            }]
                        )
                        
                        recognized_text = recognition_result['message']['content']
                        print(f"✅ 识别结果: {recognized_text}")
                        
                        # 步骤2: 数学解题
                        print("🧮 解答数学问题...")
                        
                        # 调用qwen2:latest进行数学解题
                        math_result = ollama.chat(
                            model='qwen2:latest',
                            messages=[{
                                'role': 'user',
                                'content': f"请详细解答以下数学问题：\n{recognized_text}"
                            }]
                        )
                        
                        final_answer = math_result['message']['content']
                        return [final_answer]
        
        return ["请上传包含数学题目的图片"]

# 创建智能体
solver = MathSolver()

# 使用函数包装器
def solve_math_problem(messages):
    """解题函数"""
    return solver.solve(messages)

if __name__ == "__main__":
    # 启动Web界面
    WebUI(solve_math_problem).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )