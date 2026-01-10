import json
import os

"""
1. 启动客户端
2. 链接服务端
3. 回收服务端的资源
"""
from argparse import ArgumentParser
from contextlib import AsyncExitStack
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import sys, asyncio
from dotenv import load_dotenv
load_dotenv()
import traceback
import httpx
from openai import OpenAI


class MCPClient(object):
    def __init__(self):
        self.session = None #上下文管理
        self.stdio, self.write = None, None
        self.exit_stack = AsyncExitStack()
        print(os.getenv("API_KEY"))
        print(os.getenv("BASE_URL"))
        print(os.getenv("MODEL"))
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            http_client=httpx.Client(verify=False)
        )
        self.model = os.getenv("MODEL")
        self.messages = []

    async def init(self):
        self.messages=[{
            "role": "system",
            "content": "you are a helpful assistant"
        }]
    async def cleanup(self):
        await self.exit_stack.aclose()
        print("清理完成")

    async def connect_server(self, server_script_path):
        if not server_script_path.endswith(".py"):
            ValueError("服务端的脚本必须是python文件，请先检查")

        #创建启动服务端服务的参数
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env = None,
            encoding='utf-8',
            errors='replace'
        )

        #启动服务
        self.stdio, self.write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        #调用查看都有那些工具
        response = await self.session.list_tools()
        print("链接服务器成功，服务段支持一下工具:", [tool.name for tool in response.tools])

    async def process_query(self, query):
        if not self.messages:
            await self.init()

        messages = [{"role": "user", "content": query}]
        tools_info = await self.session.list_tools()
        print("工具信息：", tools_info)

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in tools_info.tools]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools
        )

        message = response.choices[0].message
        if response.choices[0].finish_reason == "tool_calls":
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"执行的工具名: {tool_name}, 参数: {tool_args}")

            messages.append(message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id
            })
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        return message.content

    async def chat(self):
        print("exit[退出], restart[开启新一轮对话]")
        while True:
            try:
                query = input("请输入:")
                if query.lower() == "exit":
                    break
                if query.lower() == "restart":
                    await self.init()

                # 这里注释掉原本会出错的代码，暂时打印提示信息
                response = await self.process_query(query)
                print(f"结果: {response}")

            except Exception as err:
                traceback.print_stack()
                print(f"异常：{str(err)}")

parse = ArgumentParser(description=__doc__)
parse.add_argument(
    "--server_script", type=str, required = True,
)

args = parse.parse_args()
async def main():
    server_script = args.server_script
    client = MCPClient()
    try:
        print("开始启动")
        await client.connect_server(server_script)
        await client.chat()
    finally:
        await client.cleanup()

    print("over")

if __name__ == '__main__':
    asyncio.run(main())

'''
query focus summary ->graphrag
传统问答->选择rag方式， 
'''