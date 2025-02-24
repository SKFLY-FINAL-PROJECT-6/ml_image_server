from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="deepseek/deepseek-r1:free",
  messages=[
    {
      "role": "system", 
      "content": "when you get a full setence, you should break it into phrases and alot descrptive words and terms of athomsphere, emotions, and so on. you must add 2D painting style and child's drawing in the phrases generate in only 30 words only"
    }, 
    {
      "role": "user",
      "content": "generate multicular related pictures"
    }
  ]
)

print(completion.choices[0].message.content)
