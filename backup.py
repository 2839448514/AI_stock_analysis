import time
import ollama
from DrissionPage import Chromium
import re
import socket
import threading
import akshare as ak
from datetime import datetime, timedelta

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
        # 计算技术指标
        technical = calculate_technical_indicators(stock_info)
        # 分析市场情绪
        sentiment = analyze_market_sentiment(stock_info)

        base_prompt = f"""作为专业的股票分析师，请基于以下详细数据进行深入分析：

【基础行情】
股票：{stock_info.get('name', '未知')}
最新价：{stock_info.get('price', '未知')}元  |  涨跌幅：{stock_info.get('change', '未知')}%
换手率：{stock_info.get('turnover', '未知')}%  |  市盈率：{stock_info.get('pe', '未知')}

【技术指标】
5日均价：{technical.get('MA5', '未知')}
振幅：{technical.get('amplitude', '未知')}%
量比：{technical.get('volume_ratio', '未知')}

【成交信息】
成交量：{stock_info.get('volume', '未知')}  
成交额：{stock_info.get('amount', '未知')}

【K线数据】(近5日)
日期：{stock_info.get('daily_data', {}).get('dates', [])}
价格：{stock_info.get('daily_data', {}).get('prices', [])}
最高：{stock_info.get('daily_data', {}).get('highs', [])}
最低：{stock_info.get('daily_data', {}).get('lows', [])}

【资金数据】
主力净流入：{stock_info.get('fund_flow', {}).get('main_net', '未知')}
散户净流入：{stock_info.get('fund_flow', {}).get('retail_net', '未知')}

【机构评级】
覆盖机构数：{stock_info.get('ratings', {}).get('count', '未知')}
平均目标价：{stock_info.get('ratings', {}).get('avg_target', '未知')}

【市场情绪】
{' '.join(sentiment)}

用户问题：{user_input}

请从以下维度进行专业分析：
1. 技术面分析
   - 价格形态与趋势判断
   - 量价配合分析
   - 支撑与压力位判断
2. 基本面评估
   - 估值水平分析
   - 机构评级解读
3. 资金面分析
   - 主力资金动向
   - 市场情绪研判
4. 风险提示
   - 技术风险
   - 市场风险
5. 操作建议
   - 短期策略
   - 中期展望

请用专业但通俗的语言给出分析结论。"""

        # 添加50日历史数据分析
        historical_data = stock_info.get('historical_data', {})
        historical_analysis = f"""
【50日历史数据分析】
区间涨跌幅：{historical_data.get('change_rate', '未知')}%
平均价格：{historical_data.get('avg_price', '未知')}元
最高价：{historical_data.get('max_price', '未知')}元
最低价：{historical_data.get('min_price', '未知')}元
平均成交量：{historical_data.get('avg_volume', '未知')}

【价格分布】
最近50个交易日价格走势：
{historical_data.get('prices', ['未知'])}

【成交量分布】
最近50个交易日成交量走势：
{historical_data.get('volumes', ['未知'])}

请补充以下分析维度：
9. 中期趋势分析（50日）
   - 价格运行区间
   - 成交量变化特征
   - 支撑压力位判断
10. 波动特征分析
    - 振幅特征
    - 成交量特征
    - 涨跌规律
11. 历史统计分析
    - 价格分布特征
    - 成交量分布特征
    - 涨跌周期规律"""

        base_prompt = f"{base_prompt}\n{historical_analysis}"

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