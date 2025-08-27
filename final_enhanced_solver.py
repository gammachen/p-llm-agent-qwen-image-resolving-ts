import os
import ollama
import gradio as gr
import base64
from io import BytesIO
from PIL import Image
import time

def encode_pil_image(pil_image):
    """将PIL图像编码为base64字符串"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def solve_math_from_image(image):
    """数学解题函数"""
    if image is None:
        return "请上传包含数学题目的图片"
    
    try:
        # 步骤1：图像识别
        yield "🔍 正在识别图片中的数学内容...", ""
        time.sleep(0.5)
        
        base64_image = encode_pil_image(image)
        
        vision_response = ollama.chat(
            model='granite3.2-vision',
            messages=[{
                'role': 'user',
                'content': '请识别图片中的数学方程式或题目，并转换为清晰的文本格式',
                'images': [base64_image]
            }]
        )
        
        recognized_text = vision_response['message']['content']
        yield f"✅ 识别完成：\n{recognized_text}\n\n🧮 正在解答数学问题...", ""
        time.sleep(0.5)
        
        # 步骤2：数学解题
        math_response = ollama.chat(
            model='qwen2:latest',
            messages=[{
                'role': 'user',
                'content': f"请详细解答以下数学问题：\n{recognized_text}\n\n请提供完整的解题步骤和最终答案。"
            }]
        )
        
        final_answer = math_response['message']['content']
        yield "✅ 解答完成！", final_answer
        
    except Exception as e:
        yield f"❌ 解题过程中出现错误", f"错误信息：{str(e)}"

# 创建更美观的Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="数学解题智能体") as app:
    gr.Markdown("""
    # 📐 数学解题智能体
    
    上传数学题目图片，AI将自动识别并详细解答！
    
    **使用流程：**
    1. 📸 上传包含数学题目的图片
    2. 🔄 系统将自动识别数学内容
    3. 🧮 AI将提供详细解题步骤
    4. ✅ 查看完整解答结果
    
    **支持的题型：** 代数、几何、方程组、微积分等
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📸 上传数学题目图片",
                height=300
            )
            
            solve_btn = gr.Button("🚀 开始解题", variant="primary", size="lg")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                example_btn = gr.Button("📊 示例", variant="secondary")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="🔄 处理状态",
                lines=2,
                interactive=False,
                value="等待上传图片..."
            )
            
            result_output = gr.Textbox(
                label="📝 详细解答",
                lines=20,
                interactive=False,
                placeholder="解题结果将在这里显示..."
            )
    
    # 绑定事件
    solve_btn.click(
        solve_math_from_image,
        inputs=[image_input],
        outputs=[status_output, result_output]
    )
    
    clear_btn.click(
        lambda: (None, "等待上传图片...", ""),
        outputs=[image_input, status_output, result_output]
    )
    
    # 示例功能
    def load_example():
        return gr.Image("example.png"), "📸 已加载示例图片，点击开始解题按钮"
    
    example_btn.click(
        load_example,
        outputs=[image_input, status_output]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,  # 使用7863端口避免冲突
        share=False,
        favicon_path=None,
        show_error=True
    )