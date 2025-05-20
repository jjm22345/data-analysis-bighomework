import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': '',  # 填入你的Cookie（可选）
    'Host': 'store.steampowered.com',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
}

def gamename(soup):   # 游戏名字
    try:
        a = soup.find(class_="apphub_AppName")
        if a is not None:
            k = str(a.string).strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取游戏名字时出错: {e}")
        k = ""
    return k

def gameprice(soup):  # 价格
    try:
        a = soup.findAll(class_="discount_original_price")
        for i in a:
            if re.search('¥|free|免费', str(i), re.IGNORECASE):
                a = i
                break
        else:
            a = soup.findAll(class_="game_purchase_price price")
            for i in a:
                if re.search('¥|free|免费', str(i), re.IGNORECASE):
                    a = i
                    break
            else:
                a = None

        if a is not None and a.string is not None:
            k = str(a.string).replace('	', '').replace('\n', '').replace('\r', '').replace(' ', '')
        else:
            k = "未知"
    except Exception as e:
        print(f"获取价格时出错: {e}")
        k = "未知"
    return k

def taglist(soup):  # 标签列表
    list1 = []
    a = soup.find_all(class_="app_tag")
    for i in a:
        if i.string is not None:
            k = str(i.string).replace('	', '').replace('\n', '').replace('\r', '')
            if k != '+':
                list1.append(k)
    list1 = str('\n'.join(list1))
    return list1

def description(soup):  # 游戏描述
    try:
        a = soup.find(class_="game_description_snippet")
        if a is not None:
            k = str(a.text).replace('	', '').replace('\n', '').replace('\r', '').strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取游戏描述时出错: {e}")
        k = ""
    return k

def reviewsummary(soup):  # 总体评价
    try:
        a = soup.find(class_="summary column")
        if a is not None and a.span is not None:
            k = str(a.span.string).strip()
        elif a is not None:
            k = str(a.text).strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取总体评价时出错: {e}")
        k = ""
    return k

def getdate(soup):  # 发行日期
    try:
        a = soup.find(class_="date")
        if a is not None:
            k = str(a.string).strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取发行日期时出错: {e}")
        k = ""
    return k

def userreviewsrate(soup):  # 总体数量好评率
    try:
        a = soup.find(class_="user_reviews_summary_row")
        if a is not None and 'data-tooltip-html' in a.attrs:
            k = str(a.attrs['data-tooltip-html']).strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取总体数量好评率时出错: {e}")
        k = ""
    return k

def developer(soup):  # 开发商
    try:
        a = soup.find(id="developers_list")
        if a is not None and a.a is not None:
            k = str(a.a.string).strip()
        else:
            k = ""
    except Exception as e:
        print(f"获取开发商时出错: {e}")
        k = ""
    return k

def getdetail(x):
    tag, des, reviews, date, rate, dev, review, name, price = ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
    global count
    retry_count = 3  # 设置最大重试次数
    base_delay = 1  # 初始延迟时间（秒）
    while retry_count > 0:
        try:
            r = requests.get(x['Link'], headers=headers, timeout=10)
            r.raise_for_status()  # 检查请求是否成功
            soup = BeautifulSoup(r.text, 'lxml')
            name = gamename(soup)
            tag = taglist(soup)
            des = description(soup)
            reviews = reviewsummary(soup)
            date = getdate(soup)
            rate = userreviewsrate(soup)
            dev = developer(soup)
            price = gameprice(soup)
            print(f'已完成: {name}{str(x["ID"])} 第{count}个')
            break  # 成功后退出循环
        except requests.exceptions.ConnectionError as e:
            print(f'服务器无响应，正在重试... ({retry_count}次剩余)')
            retry_count -= 1
            delay = base_delay * (2 ** (3 - retry_count))  # 指数退避算法
            print(f'等待 {delay} 秒后重试...')
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"请求出错: {e}")
            retry_count -= 1
            delay = base_delay * (2 ** (3 - retry_count))  # 指数退避算法
            print(f'等待 {delay} 秒后重试...')
            time.sleep(delay)
        except Exception as e:
            print(f'未完成: {str(x["ID"])} 第{count}个, 错误信息: {e}')
            price = 'error'
            break
    count += 1
    return name, price, tag, des, reviews, date, rate, dev, ''

if __name__ == "__main__":
    df1 = pd.read_excel('1.xlsx')
    count = 1
    df1['详细'] = df1.apply(lambda x: getdetail(x), axis=1)
    df1['名字'] = df1.apply(lambda x: x['详细'][0], axis=1)
    df1['价格'] = df1.apply(lambda x: x['详细'][1], axis=1)
    df1['标签'] = df1.apply(lambda x: x['详细'][2], axis=1)
    df1['描述'] = df1.apply(lambda x: x['详细'][3], axis=1)
    df1['近期评价'] = df1.apply(lambda x: x['详细'][4], axis=1)
    df1['发行日期'] = df1.apply(lambda x: x['详细'][5], axis=1)
    df1['近期数量好评率'] = df1.apply(lambda x: x['详细'][6], axis=1)
    df1['开发商'] = df1.apply(lambda x: x['详细'][7], axis=1)
    
    df1.to_excel('2.xlsx')
    print('已完成全部')