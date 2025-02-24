from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

user_prompt = "수풀이 우거진 숲 그림을 그려줘"
system_prompt = {
    'enhancing' : 'You are a prompt engineer. Process the following Korean prompt by translating it into English and extracting key descriptive terms (nouns and adjectives). Only output the extracted words in a single line, separated by commas. Do not include any explanations or translations.',
    'natural' : 'Translate this sentence into English.'
}

mode = 'enhancing'

completion = client.chat.completions.create(
  # 사용 모델 : mistralai/mistral-7b-instruct
  model = "mistralai/mistral-7b-instruct",
  messages=[
    {
      # 역할 부여용 프롬프트
      "role": "system", 
      "content": system_prompt[mode]
    },
    {
      # 유저에게 입력받은 프롬프트가 들어가는 부분 
      "role": "user",
      "content": user_prompt
    }
  ]
)

# 최종 결과물 출력
print(completion.choices[0].message.content)
