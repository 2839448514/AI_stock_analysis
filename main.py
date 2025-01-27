import time
import ollama
from DrissionPage import Chromium
import re
import socket
import threading
import akshare as ak
from datetime import datetime, timedelta
import numpy as np

messages = [{'role': 'user', 'content': "你是一个股票助手，请用简短的语言回答问题"}]


def get_stock_info(stock_code):
    """获取更全面的股票信息"""
    try:
        info = {}
        # 获取实时行情
        df_real = ak.stock_zh_a_spot_em()
        stock_real = df_real[df_real['代码'] == stock_code]
        if not stock_real.empty:
            info['name'] = stock_real['名称'].values[0]
            info['price'] = stock_real['最新价'].values[0]
            info['change'] = stock_real['涨跌幅'].values[0]
            info['volume'] = stock_real['成交量'].values[0]
            info['amount'] = stock_real['成交额'].values[0]
            info['turnover'] = stock_real['换手率'].values[0]
            info['pe'] = stock_real['市盈率-动态'].values[0]

        # 获取50日K线数据
        df_hist = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily",
            start_date=(datetime.now() - timedelta(days=70)).strftime('%Y%m%d'),
            adjust="qfq"
        )
        if not df_hist.empty:
            recent50 = df_hist.tail(50)
            recent5 = df_hist.tail(5)
            
            # 5日数据
            info['daily_data'] = {
                'dates': recent5['日期'].tolist(),
                'prices': recent5['收盘'].tolist(),
                'volumes': recent5['成交量'].tolist(),
                'highs': recent5['最高'].tolist(),
                'lows': recent5['最低'].tolist()
            }
            
            # 50日数据
            info['historical_data'] = {
                'dates': recent50['日期'].tolist(),
                'prices': recent50['收盘'].tolist(),
                'volumes': recent50['成交量'].tolist(),
                'highs': recent50['最高'].tolist(),
                'lows': recent50['最低'].tolist(),
                'avg_price': recent50['收盘'].mean(),
                'max_price': recent50['最高'].max(),
                'min_price': recent50['最低'].min(),
                'avg_volume': recent50['成交量'].mean()
            }
            
            # 计算区间涨跌幅
            start_price = recent50['收盘'].iloc[0]
            end_price = recent50['收盘'].iloc[-1]
            info['historical_data']['change_rate'] = ((end_price - start_price) / start_price) * 100

        # 获取主力资金流向
        try:
            df_flow = ak.stock_individual_fund_flow(stock=stock_code)
            if not df_flow.empty:
                info['fund_flow'] = {
                    'main_net': df_flow['主力净流入'].values[0],
                    'retail_net': df_flow['小单净流入'].values[0]
                }
        except:
            pass

        # 获取机构评级
        try:
            df_rating = ak.stock_rank_forecast_cninfo(symbol=stock_code)
            if not df_rating.empty:
                info['ratings'] = {
                    'count': len(df_rating),
                    'avg_target': df_rating['目标价'].mean()
                }
        except:
            pass

        return info
    except Exception as e:
        return {'error': str(e)}


def analyze_market_sentiment(stock_info):
    """分析市场情绪"""
    sentiment = []
    try:
        # 分析换手率
        turnover = float(stock_info.get('turnover', 0))
        if turnover > 5:
            sentiment.append("换手率较高，市场交投活跃")
        elif turnover < 2:
            sentiment.append("换手率偏低，交易相对清淡")

        # 分析资金流向
        if 'fund_flow' in stock_info:
            main_net = float(stock_info['fund_flow']['main_net'])
            retail_net = float(stock_info['fund_flow']['retail_net'])
            if main_net > 0:
                sentiment.append("主力资金呈净流入状态")
            else:
                sentiment.append("主力资金呈净流出状态")

        # 分析价格变动
        if 'daily_data' in stock_info:
            prices = stock_info['daily_data']['prices']
            if len(prices) >= 2:
                price_change = ((prices[-1] - prices[0]) / prices[0]) * 100
                if abs(price_change) > 5:
                    sentiment.append(f"近期股价波动较大，变动幅度{price_change:.2f}%")

    except Exception:
        pass
    return sentiment


def calculate_technical_indicators(stock_info):
    """计算技术指标"""
    indicators = {}
    try:
        if 'daily_data' in stock_info:
            prices = stock_info['daily_data']['prices']
            volumes = stock_info['daily_data']['volumes']

            # 计算MA5
            if len(prices) >= 5:
                indicators['MA5'] = sum(prices) / len(prices)

            # 计算量比
            if len(volumes) >= 2:
                vol_ratio = volumes[-1] / sum(volumes[:-1]) * (len(volumes) - 1)
                indicators['volume_ratio'] = vol_ratio

            # 计算振幅
            if 'highs' in stock_info['daily_data'] and 'lows' in stock_info['daily_data']:
                amplitude = ((max(stock_info['daily_data']['highs']) -
                              min(stock_info['daily_data']['lows'])) /
                             prices[0] * 100)
                indicators['amplitude'] = amplitude

    except Exception:
        pass
    return indicators


def generate_prompt(user_input, stock_info=None):
    """生成更专业的AI提示"""
    if stock_info and 'error' not in stock_info:
        # 获取基础数据
        technical = calculate_technical_indicators(stock_info)
        sentiment = analyze_market_sentiment(stock_info)
        historical_data = stock_info.get('historical_data', {})

        base_prompt = f"""作为专业的股票分析师，请基于以下数据进行深入分析并给出具体建议：

【基础行情与技术面】
股票：{stock_info.get('name', '未知')}({stock_info.get('code', '未知')})
最新价：{stock_info.get('price', '未知')}元
涨跌幅：{stock_info.get('change', '未知')}%
换手率：{stock_info.get('turnover', '未知')}%
市盈率：{stock_info.get('pe', '未知')}
5日均价：{technical.get('MA5', '未知')}
50日均价：{historical_data.get('avg_price', '未知')}元

【历史区间】(50日数据)
最高价：{historical_data.get('max_price', '未知')}元
最低价：{historical_data.get('min_price', '未知')}元
区间涨跌幅：{historical_data.get('change_rate', '未知')}%
价格走势：{historical_data.get('prices', ['未知'])[-10:]}（显示最近10日）

【市场资金】
主力净流入：{stock_info.get('fund_flow', {}).get('main_net', '未知')}
散户净流入：{stock_info.get('fund_flow', {}).get('retail_net', '未知')}
市场情绪：{' '.join(sentiment)}

【机构评级】
覆盖机构数：{stock_info.get('ratings', {}).get('count', '未知')}
平均目标价：{stock_info.get('ratings', {}).get('avg_target', '未知')}

用户问题：{user_input}

请基于以上数据，从专业角度给出以下分析：

1. 技术面研判
   - 目前处于什么位置（相对高低位）
   - 支撑位和压力位在哪里，依据是什么
   - 短期趋势研判

2. 交易建议
   - 给出明确的买入区间建议价位
   - 给出目标价和止损价
   - 说明建议持仓时间
   - 建议仓位配置
   
3. 风险提示
   - 列出主要风险点
   - 哪些情况要立即止损

请用专业但通俗的语言回答，给出明确的数字建议。
所有建议均应该基于专业分析，而不是机械计算。
注意：建议仅供参考，投资者需承担风险。"""

    else:
        base_prompt = f"""作为专业的股票分析师，请回答以下问题：{user_input}
如果涉及具体股票，请说明缺乏实时数据，无法进行具体分析。"""

    return base_prompt


def ask_ollama(ask):
    try:
        messages.append({'role': 'user', 'content': ask})
        response = ollama.chat(model='deepseek-r1:14b', messages=messages)
        assistant_reply = response["message"]["content"]
        messages.append({'role': 'assistant', 'content': assistant_reply})
        return assistant_reply
    except Exception as e:
        return f"查询失败: {str(e)}"


def format_stock_code(code):
    """格式化股票代码"""
    # 移除可能的 sh/sz 前缀
    code = re.sub(r'^[sh|sz|SH|SZ]+', '', code)

    # 根据首位数字判断添加正确前缀
    if code.startswith('6'):
        return code  # akshare查询时不需要sh前缀
    elif code.startswith(('0', '3')):
        return code  # akshare查询时不需要sz前缀
    return None


def main():
    print("AI股票助手已启动(输入'退出'结束程序)")
    while True:
        user_input = input("\n请输入问题: ")
        if user_input.lower() == '退出':
            break

        stock_match = re.search(r'[sh|sz|SH|SZ]?[036]\d{5}', user_input)

        if stock_match:
            code = format_stock_code(stock_match.group())
            if code:
                stock_info = get_stock_info(code)
                prompt = generate_prompt(user_input, stock_info if 'error' not in stock_info else None)
            else:
                prompt = generate_prompt(f"用户输入了无效的股票代码。{user_input}")
        else:
            prompt = generate_prompt(user_input)

        result = ask_ollama(prompt)
        print(f"Assistant: {result}")


if __name__ == "__main__":
    main()