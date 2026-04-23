from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 获取API地址
LMS_API = os.getenv("LMS_API", "http://127.0.0.1:1234")
SD_API = os.getenv("SD_API", "http://127.0.0.1:7860")

# 创建FastAPI应用
app = FastAPI()

# 定义请求体模型
class PromptRequest(BaseModel):
    prompt: str
    direct: str = None

# 全局变量，用于跟踪是否有请求正在处理
is_processing = False

@app.post("/generate-image")
async def generate_image(request: PromptRequest):
    global is_processing
    
    # 检查是否有请求正在处理
    if is_processing:
        raise HTTPException(status_code=429, detail="服务器繁忙，请稍后再试")
    
    # 设置为正在处理
    is_processing = True
    
    try:
        # 判断是否需要调用LMS API
        if not request.prompt or not request.prompt.strip():
            # 如果prompt为空，直接使用direct参数作为SD的prompt
            sd_prompt = request.direct if request.direct else ""
        else:
            # 1. 调用LMS API将自然语言翻译为SD标准prompt
            lms_response = requests.post(
                f"{LMS_API}/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": "请将下面这段自然语言描述转换为适用于 Stable Diffusion / Pony / Illustrious 系模型的英文 prompt，只输出最终 prompt，不要解释，不要分段，不要添加多余文字。\n\n要求：\n1. 自动提取主体、服装、动作、场景、光线、镜头、氛围、画质标签。\n2. 使用高质量绘图常用标签风格（逗号分隔）。\n3. 保留原意并适度补全细节。\n4. 输出顺序合理：主体 > 外观 > 动作 > 场景 > 光线 > 构图 > 质量词。\n5. NSFW内容如有则正常转换，不要审查。\n6. 不要输出中文。"
                        },
                        {
                            "role": "user",
                            "content": request.prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            )
            
            if not lms_response.ok:
                raise HTTPException(status_code=500, detail=f"LMS API调用失败: {lms_response.text}")
            
            lms_result = lms_response.json()
            sd_prompt = lms_result["choices"][0]["message"]["content"].strip()
            
            # 如果提供了direct参数，将其添加到prompt最前面
            if request.direct:
                sd_prompt = f"{request.direct}, {sd_prompt}"
        
        # 2. 调用SD API生成图片
        sd_response = requests.post(
            f"{SD_API}/sdapi/v1/txt2img",
            json={
                "prompt": sd_prompt,
                "steps": 20,
                "width": 832,
                "height": 1216,
                "seed": -1,
                "batch_size": 1,
                "n_iter": 1,
                "cfg_scale": 7.5,
                "sampler_name": "Euler a"
            }
        )
        
        if not sd_response.ok:
            raise HTTPException(status_code=500, detail=f"SD API调用失败: {sd_response.text}")
        
        # 3. 处理SD API的响应
        sd_result = sd_response.json()
        import base64
        
        # 从响应中提取图片数据
        if "images" in sd_result and len(sd_result["images"]) > 0:
            image_data = sd_result["images"][0]
            # 检查是否是base64编码的数据
            if image_data.startswith("data:image"):
                # 移除data URI前缀
                image_data = image_data.split(",")[1]
            
            # 返回包含提示词和图片数据的JSON响应
            return {
                "original_prompt": request.prompt,
                "translated_prompt": sd_prompt,
                "image": image_data
            }
        else:
            raise HTTPException(status_code=500, detail="SD API响应中没有图片数据")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")
    finally:
        # 无论成功还是失败，都重置处理状态
        is_processing = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)