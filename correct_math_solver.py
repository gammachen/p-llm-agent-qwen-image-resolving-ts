import os
import ollama
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import encode_image_as_base64

def math_solver(messages):
    """数学解题智能体主函数"""
    if not messages:
        yield ["请上传包含数学题目的图片"]
        return
    
    # 获取最后一条消息
    last_msg = messages[-1]
    
    # 检查是否有图片
    image_data = None
    
    try:
        if hasattr(last_msg, 'content') and last_msg.content:
            content = last_msg.content
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, 'image') and item.image:
                        image_path = str(item.image).replace('file://', '')
                        
                        if os.path.exists(image_path):
                            image_data = encode_image_as_base64(image_path)
                        else:
                            image_data = image_path
                        break
            elif isinstance(content, str) and content.startswith('file://'):
                image_path = content.replace('file://', '')
                if os.path.exists(image_path):
                    image_data = encode_image_as_base64(image_path)
                else:
                    image_data = image_path
    except Exception as e:
        print(f"处理图片时出错: {e}")
    
    if not image_data:
        yield ["请上传包含数学题目的图片"]
        return
    
    try:
        print("🔍 步骤1: 识别图片中的数学内容...")
        
        # 步骤1: 图像识别
        vision_response = ollama.chat(
            model='granite3.2-vision',
            messages=[{
                'role': 'user',
                'content': '请识别图片中的数学方程式或题目，并转换为清晰的文本格式',
                'images': [image_data]
            }]
        )
        
        recognized_text = vision_response['message']['content']
        print(f"✅ 识别结果: {recognized_text}")
        
        print("🧮 步骤2: 解答数学问题...")
        
        # 步骤2: 数学解题
        math_response = ollama.chat(
            model='qwen2:latest',
            messages=[{
                'role': 'user',
                'content': f"请详细解答以下数学问题：\n{recognized_text}\n\n请提供完整的解题步骤和最终答案。"
            }]
        )
        
        final_answer = math_response['message']['content']
        yield [final_answer]
        
    except Exception as e:
        error_msg = f"解题过程中出现错误: {str(e)}"
        print(error_msg)
        yield [error_msg]

# 创建智能体配置
agent_config = {
    'name': '数学解题智能体',
    'description': '上传数学题目图片，AI将识别并解答',
    'function': math_solver
}

if __name__ == "__main__":
    # 启动Web界面
    WebUI([agent_config]).run(
        server_name="0.0.0.0",
        server_port=7862,
        title="数学解题智能体",
        description="上传数学题目图片，AI将识别并解答"
    )