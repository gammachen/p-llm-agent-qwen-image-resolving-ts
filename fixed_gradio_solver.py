import os
import ollama
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

def encode_pil_image(pil_image):
    """将PIL图像编码为base64字符串"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def solve_math_from_image(image):
    """数学解题函数"""
    if image is None:
        return "请上传图片"
    
    try:
        print("🔍 步骤1: 识别图片中的数学内容...")
        
        # 将PIL图像编码为base64
        base64_image = encode_pil_image(image)
        
        # 步骤1: 图像识别
        vision_response = ollama.chat(
            model='granite3.2-vision',
            messages=[{
                'role': 'user',
                'content': '请识别图片中的数学方程式或题目，并转换为清晰的文本格式',
                'images': [base64_image]
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
        return final_answer
        
    except Exception as e:
        error_msg = f"解题过程中出现错误: {str(e)}"
        print(error_msg)
        return error_msg

# 创建Gradio界面
interface = gr.Interface(
    fn=solve_math_from_image,
    inputs=gr.Image(type="pil", label="上传数学题目图片"),
    outputs=gr.Textbox(label="解题结果", lines=15),
    title="数学解题智能体",
    description="上传数学题目图片，AI将使用granite3.2-vision识别内容，然后使用qwen2:latest解答",
    examples=None,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # 启动Gradio界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )