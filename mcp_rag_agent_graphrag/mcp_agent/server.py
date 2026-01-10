import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx
import json
#创建一个对象
mcp = FastMCP("weather")


async def get_weather(city: str) -> dict[str, Any]:
    """
    :param city:
    :return:
    """
    """
        1.确定查询天气的服务
        2.配置相关的参数，构建请求体
        3.请求服务
       api.weatherapi.com
    """
    #查询当前时间的天气情况
    url = "http://api.weatherapi.com/v1/current.json"
    api_key=""
    parameter = {
        "q": city,
        "key": api_key
    }
    #异步方式来请求
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url,params=parameter)
            return response.json()
        except Exception as err:
            return {"error": f"请求失败: {str(err)}"}

async def format_data(json_data):
    """
    将天气数据转成已读文本
    :param json_data:
    :return:
    """
    """
    {'location': {'name': 'Shenzhen', 'region': 'Guangdong', 'country': 'China', 'lat': 22.5333, 'lon': 114.1333, 'tz_id': 'Asia/Hong_Kong', 'localtime_epoch': 1745491061, 'localtime': '2025-04-24 18:37'}, 'current': {'last_updated_epoch': 1745490600, 'last_updated': '2025-04-24 18:30', 'temp_c': 27.1, 'temp_f': 80.8, 'is_day': 1, 'condition': {'text': 'Moderate or heavy rain with thunder', 'icon': '//cdn.weatherapi.com/weather/64x64/day/389.png', 'code': 1276}, 'wind_mph': 9.8, 'wind_kph': 15.8, 'wind_degree': 186, 'wind_dir': 'S', 'pressure_mb': 1007.0, 'pressure_in': 29.74, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 89, 'cloud': 75, 'feelslike_c': 29.5, 'feelslike_f': 85.2, 'windchill_c': 27.3, 'windchill_f': 81.2, 'heatindex_c': 29.9, 'heatindex_f': 85.8, 'dewpoint_c': 21.7, 'dewpoint_f': 71.1, 'vis_km': 10.0, 'vis_miles': 6.0, 'uv': 0.2, 'gust_mph': 12.1, 'gust_kph': 19.5}}

    """
    city = json_data.get("location", {}).get("name", "未知")
    country = json_data.get("location", {}).get("country", "未知")
    temp = json_data.get("current", {}).get("temp_c", "N/A")
    humidity = json_data.get("current", {}).get("humidity", "N/A")
    wind_speed = json_data.get("current", {}).get("wind_kph", "N/A")
    condition = json_data.get("current", {}).get("condition", {}).get("text", "未知")
    return f"当前城市{country}.{city}\n 温度:{temp}，湿度:{humidity}，风速:{wind_speed}， 天气情况:{condition}"

#定义成一个工具
@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定的城市名(英文)，返回今日天气情况
    :param city: 城市名称(需使用英文)
    :return: 格式化之后的天气信息
    """
    data = await get_weather(city)
    weather_info = await format_data(json_data=data)
    print("weather_info:", weather_info)
    return weather_info

import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
import httpx
import json
#创建一个对象
mcp = FastMCP("graphrag")

if __name__=="__main__":
    asyncio.run(query_weather("Shenzhen"))
    mcp.run(transport="stdio")
'''
if __name__ == "__main__":
    # asyncio.run(query_weather("shenzhen"))
    mcp.run(transport="stdio")

    # server和client都在同一个服务器上，std IO流进行交互，提高交互效率
    # 如果server端部署在远程，那需要使用sse的方式
'''