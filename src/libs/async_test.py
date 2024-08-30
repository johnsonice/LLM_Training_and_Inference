#### openai async test

import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
env_path = '../../.env'
load_dotenv(dotenv_path=env_path)

class GPT:
    def __init__(self,model_name='gpt-4o-mini',Async=False,api_key=None):

        self.model_name = model_name
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if Async:
            print('load async client')
            self.client = AsyncOpenAI(
                api_key=api_key,
            )
        else:
            print('load sync client')
            self.client = OpenAI(
                api_key=api_key,
            )
            
    def response(self, user_prompt, 
                 sys_prompt="You are a helpful assistant.",
                 model_name=None):
        if model_name is None:
            model_name = self.model_name
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response
    
async def process_messages(messages):
    llm = GPT(model_name='gpt-4o-mini',Async=True)
    for message in messages:
        response = await llm.response(message)
        print(response.choices[0].message.content)


async def main() -> None:
    messages_list = ["Please generate a 1000 word paragraph",
                    "Say this is a test 2",
                    "Say this is a test 3",
                    "Say this is a test 4",
                    "Say this is a test 5",
                    "Say this is a test 6",
                    "Say this is a test 7",
                    "Say this is a test 8",
                    "Say this is a test 9",
                    "Say this is a test 10",
                    ]

    await process_messages(messages_list)



if __name__ == "__main__":
    
    # ### sync call
    llm_test = GPT(model_name='gpt-4o-mini',Async=False)
    print(llm_test.response(user_prompt='produce a random sentence').choices[0].message.content)
    
    ### async call
    asyncio.run(main())

