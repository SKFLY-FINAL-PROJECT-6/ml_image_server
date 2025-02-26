import os

from openai import OpenAI

class LLMModel:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.system_prompt = {
            'enhancing': 'You are a prompt engineer. Process the following Korean prompt by translating it into English and extracting key descriptive terms (nouns and adjectives). Only output the extracted words in a single line, separated by commas. Do not include any explanations or translations.',
            'natural': 'Translate this sentence into English.'
        }

    def process_prompt(self, user_prompt: str, mode: str = 'enhancing') -> str:

        completion = self.client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt[mode]
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ]
        )
        
        return completion.choices[0].message.content