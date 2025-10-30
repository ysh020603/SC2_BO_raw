import os
import json
import requests
import openai

def query_deepseek(prompt):
    try:
        # response = requests.post(
        #     url="https://openrouter.ai/api/v1/chat/completions",
        #     headers={
        #         "Authorization": f"Bearer {API_KEY}",
        #         "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #         "X-Title": "<YOUR_SITE_NAME>",
        #     },
        #     data=json.dumps({
        #         "model": "deepseek/deepseek-r1-0528:free", # Optional
        #         "messages": [
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #         ]
        #     })
        # )
        client = openai.Client(
            base_url="http://172.18.132.21:30000/v1", api_key="EMPTY"
        )

        # response = client.chat.completions.create(
        response = client.chat.completions.create(
            model="DeepSeek-R1",
            messages=[
                # {"role": "system", "content": "You are a helpful AI assistant"},
                # {"role": "user", "content": prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
            stream=False  # Enable streaming
        )

        # response.raise_for_status()  # 检查HTTP错误
        
        # result = response.json()
        return response.choices[0].message.content
    
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"
    except KeyError:
        return "解析响应失败，请检查API返回结构"

def translate(prompt):
    prompt = f"以下是对星际争霸策略相关的描述。请将该描述忠实地翻译为英文，使其流畅连贯，你的输出只能包含翻译之后的结果，不需要其它的内容：{prompt}"
    ans = query_deepseek(prompt)
    ans = ans.split('</think>\n')[-1]
    return ans


def read_all_json_files(folder_path):
    """
    读取指定文件夹中的所有 JSON 文件，并返回一个字典，
    键为文件名，值为解析后的 JSON 数据。
    """
    json_data = {}
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return json_data

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只处理文件且是 .json 扩展名
        if os.path.isfile(file_path) and filename.lower().endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    json_data[filename] = data
                    print(f"成功读取: {filename}")
            except Exception as e:
                print(f"读取或解析失败 {filename}: {e}")
    
    return json_data

def find_bo(bo, sub_bo):
    start = None
    end = None

    sub_start = sub_bo[0]
    sub_end = sub_bo[-1]
    
    for i in range(len(bo)):
        if sub_start[0] == bo[i][0] and sub_start[1] == bo[i][1] and sub_start[2] == bo[i][2]:
            start = i
        if sub_end[0] == bo[i][0] and sub_end[1] == bo[i][1] and sub_end[2] == bo[i][2]:
            end = i
            break
    
    if start == None or end == None:
        return []
    else:
        return bo[start:end+1]

# 使用示例
if __name__ == "__main__":
    error_ct = 0
    folder = r"/data/shy/RL_LLM/SC2Arena/BO/strategies_ann_train"  # 替换为你的文件夹路径
    all_json = read_all_json_files(folder)
    
    bo_folder = r"/data/shy/RL_LLM/SC2Arena/BO/build_order"
    all_bo = os.listdir(bo_folder)

    prompts = []
    
    # 打印结果示例
    for fname, content in all_json.items():
        with open(f"{bo_folder}/{fname}", 'r') as f:
            bo = json.load(f)

        hexin = translate(content["核心思想与战术目标"])
        
        history = "暂无。"
        
        for unit_build_order in content["阶段划分"]:
            stage = translate(unit_build_order["阶段"])
            aim = translate(unit_build_order["目标"])
            action = translate(unit_build_order["关键操作"])
            reasoning = translate(unit_build_order["reasoning"])

            del unit_build_order["目标"]
            del unit_build_order["关键操作"]

            res = find_bo(bo, unit_build_order['BO切片'])
            
            # prompt_leadin = "At this stage, we provide you with recommendations on Build Orders for stage-specific decisions. Please refer to these Build Order suggestions as much as possible. \
# The time provided are only for reference, you could not follow them strictly. But do remember build all necessary builds in the BO when possible. "
            
            # prompt_aim = f"You are now at stage: {stage}. The aim for this stage: {aim}. Your operations in this stage may include: {action}. Reasoning: {reasoning}"
            # prompt_aim = f"You are now at stage: {stage}. The aim for this stage: {aim}. Your operations in this stage may include: {action}. "

            # prompt_bo = "Now, please follwing the follow build order and complete the above stage-specific decisions. Please note that this is a periodic strategy, which means that it will not change immediately until the whole BO is finished, and several builds you have done may still be in this BO. Here are the instructions for buildings: "

            for p in res:
                prompt_bo += f"Please build {p[2]} at {p[1]}. "
            
            prompt = prompt_leadin + prompt_aim + prompt_bo

            print(prompt)

            prompts.append({
                'prompt': prompt,
                'bo': res
            })
    
    with open('/data/shy/RL_LLM/SC2Arena/BO/prompts.json', 'w') as f:
        json.dump(prompts, f, indent=2)