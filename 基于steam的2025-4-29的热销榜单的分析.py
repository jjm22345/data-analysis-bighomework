import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 200)  # 设置打印宽度
plt.rcParams['font.sans-serif']=['Microsoft YaHei']#字体

class steam_ana(object):
    def __init__(self, path: str):
        self.steam_data = pd.read_excel(path)
        # 处理好评率一列
        def get_goods(x):
            x1 = x['近期数量好评率']
            if pd.isna(x1) or not isinstance(x1, str):
                return '0%'
            match1 = re.search(r'过去 30 天内的 (.*?) 篇用户评测中有 (\d*%) 为好评。', x1)
            match2 = re.search(r'(\d*) 篇用户的游戏评测中有 (\d*%) 为好评。', x1)

            if match1:
                percent = match1.group(2) 
            elif match2:
                percent = match2.group(2)
            else:
                percent = '0%'
            return percent
        # 处理价格列
        def price(x):
            price_str = x['价格']
            if '未知' in price_str:
                return float('NaN')  # 未知情况返回 NaN
            price_str = price_str.strip()
            if '免费开玩' in price_str:
                return 0  # 免费开玩替换为 0
            # 提取价格数字
            try:
                numeric_price = re.search(r'\d+', price_str.replace('¥', '').replace('$', '')).group()
                return int(numeric_price)
            except:
                return float('NaN')  # 其他无法识别的情况返回 NaN
        self.steam_data['价格'] = self.steam_data.apply(lambda x: price(x), axis=1)
        self.steam_data['价格'] = pd.to_numeric(self.steam_data['价格'])  # 转为int64
        self.steam_data['好评率'] = self.steam_data.apply(lambda x: get_goods(x), axis=1)
        #处理日期列
        self.steam_data['发行日期'] = pd.to_datetime(self.steam_data['发行日期'], format='%Y 年 %m 月 %d 日')
        # 删去对数据分析没有意义的列
        self.steam_data = self.steam_data.drop(labels=['Link', 'ID', '详细'], axis=1)
        print(f"show_data:\n{self.steam_data.sample(10)}\n\n")
        print(f"steam_data结构:\n")
        self.steam_data.info()
        print(f"价格数据的各项计算数值:\n{self.steam_data.describe()['价格']}")

    def lose_show(self):
        # 计算每列的非空值数量和缺失值数量
        count_values = self.steam_data.count()
        null_values = self.steam_data.isnull().sum()
        missing_data = pd.DataFrame({
            '总样本数': count_values,
            '缺失值': null_values
        })

        print(f"缺失数据:\n{missing_data}")

    def fill_missing_with_rf(self, column, feature_columns):
        """
        填补缺失值，使用指定的 feature_columns 作为特征
        :param column: 要填补的目标列名（如 '价格'）
        :param feature_columns: 用于预测该列的特征列名列表（如 ['好评率']）
        """
        df = self.steam_data.copy()
        target = df[column]
        # 找出缺失值位置
        nan_indices = target[target.isnull()].index
        known_indices = target[target.notnull()].index
        # 使用指定的 feature_columns 作为特征
        features = df[feature_columns]
        X_known = features.loc[known_indices]
        y_known = target.loc[known_indices]
        X_unknown = features.loc[nan_indices]
        # 训练随机森林模型
        rf = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
        rf.fit(X_known, y_known)
        # 预测并填充缺失值
        predicted = rf.predict(X_unknown)
        df.loc[nan_indices, column] = predicted
        return df

    def lose_solve(self):  # 处理缺失值
        self.steam_data['好评率'] = pd.to_numeric(
            self.steam_data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        self.steam_data = self.fill_missing_with_rf('价格', feature_columns=['好评率'])
        self.steam_data['好评率'] = self.steam_data['好评率'].apply(lambda x: f'{x:.2f}%')

    def __show_amount_types(self):  # 绘制各个游戏类型数量的直方图
        '''
            绘制前五十最多的游戏标签类型
        '''
        tags = self.steam_data['标签'].astype(str).to_list()
        all_tags = []
        for tag_str in tags:
            if tag_str.strip() == '' or tag_str == 'nan':
                continue  
            tmp_list = tag_str.split('\n')  
            all_tags.extend(tmp_list)
        frequnt = {}
        for tag in all_tags:
            if tag not in frequnt:
                frequnt[tag] = 1
            else :
                frequnt[tag] += 1
        frequnt = sorted(frequnt.items(),key = lambda x:x[1],reverse=True)
        frequnt_df = pd.DataFrame(frequnt,columns=['类型','数量'])
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        sns.barplot(data=frequnt_df.head(50),x='类型',y='数量')
        plt.xlabel('类型')
        plt.ylabel('数量')
        plt.ylim(0,150)
        plt.xticks(fontsize = 10,rotation = 60)
        plt.savefig("各个游戏类型数量的直方图.svg")
        plt.show()
        

    def __show_price(self):  # 绘制价格-发布时间分布散点图
        '''
        查看热销榜上前222款游戏的价格-发布时间分布情况
        '''
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        sns.scatterplot(data=self.steam_data, x='发行日期', y='价格',alpha=0.8)
        plt.xlabel('年份', fontsize=20)
        plt.ylabel('价格', fontsize=20)
        plt.title('热销榜上前222款游戏的价格-发布时间分布情况', fontsize=24)
        dstart = datetime.datetime(2010, 1, 1)
        dend = datetime.datetime(2026, 1, 1)
        plt.xlim(dstart, dend)     
        if not plt.show():
            print("图片生成成功")
            if not plt.savefig('热销榜上前223款游戏的价格-发布时间分布情况.svg'):
                print("图形保存成功")
            else :
                print("保存失败")
        else :
            print("图片生成失败")
        

    def __show_goodcpmment(self):  # 绘制游戏好评率-发布时间的分布散点图
        '''
        查看热销榜上前222款游戏的游戏好评率-发布时间分布情况
        
        '''
        self.steam_data['好评率'] = pd.to_numeric(
            self.steam_data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        tmp = self.steam_data[self.steam_data['好评率'] > 0]
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        sns.scatterplot(data=tmp, x='发行日期', y='好评率',alpha=0.8)
        plt.xlabel('年份', fontsize=20)
        plt.ylabel('好评率', fontsize=20)
        plt.title('热销榜上前223款游戏的好评率-发布时间分布情况', fontsize=24)
        dstart = datetime.datetime(2010, 1, 1)
        dend = datetime.datetime(2025, 12, 31)
        plt.xlim(dstart, dend)
        try:
            plt.savefig('热销榜上前223款游戏的好评率-发布时间分布情况.svg')
            print("图形保存成功")
        except Exception as e:
            print(f"保存失败: {e}")
        self.steam_data['好评率'] = self.steam_data['好评率'].apply(lambda x: f'{x:.2f}%')
        plt.show()

    def __show_years_goods_price(self):  # 绘制以发布年份分组进行平均价格和平均好评率的绘制 ———图形应该要是两条折线
        self.steam_data['好评率'] = pd.to_numeric(
            self.steam_data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        # 提取年份
        self.steam_data['年份'] = self.steam_data['发行日期'].dt.year
        # 按年份分组计算平均值
        df_yearprice = self.steam_data.groupby('年份')['价格'].mean().to_frame().reset_index().sort_values(by='年份')
        df_yearreview = self.steam_data.groupby('年份')['好评率'].mean().to_frame().reset_index().sort_values(by='年份')
        plt.figure(figsize=(20, 5))
        sns.lineplot(data=df_yearprice, x='年份', y='价格', color='green', label='平均价格')
        sns.lineplot(data=df_yearreview, x='年份', y='好评率', color='blue', label='平均好评率')
        plt.xlabel('年份', fontsize=20)
        plt.legend()
        plt.title('年份与价格、好评率')
        plt.xlim(2010, 2025)
        plt.ylim(0, 200)
        plt.savefig('以发布年份分组进行平均价格和平均好评率.svg')
        plt.show()
        self.steam_data['好评率'] = self.steam_data['好评率'].apply(lambda x: f'{x:.2f}%')
        print(df_yearprice)
        
    def draw_datas(self):
        self.__show_price()
        self.__show_goodcpmment()
        self.__show_years_goods_price()
        self.__show_amount_types()
        
    
    '''
        单变量分析
    '''
    

    def ana_goods_rate(self, phrase: str = None) -> str:
        '''
            各种好评率挡位在榜单上所有游戏的占比
        '''
        data = self.steam_data
        grouped = data['近期评价'].groupby(data['近期评价']).count()
        print(f"评价为 {phrase} 在223款游戏中的占比为: {(grouped[phrase]/data.shape[0])*100:.2f}%")
        plt.figure(figsize=(15, 10))
        # 根据数值大小生成渐变颜色
        norm = plt.Normalize(grouped.min(), grouped.max())
        colors = sns.color_palette("viridis", len(grouped))
        colors = [colors[i] for i in np.argsort(grouped.values)]
        norm = Normalize(vmin=grouped.min(), vmax=grouped.max())
        sm = ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca())
        grouped.plot(kind='bar', color=colors)
        plt.xticks(fontsize=10, rotation=60)
        plt.tight_layout()
        plt.savefig("各种评价.svg")

    def univariate_year_analysis(self): #发行年份的单变量分析
        data = self.steam_data.copy()
        data = data.dropna(subset=['发行日期'])
        data['年份'] = data['发行日期'].dt.year
        data['年份'] = data['年份'].astype(int)
        year_counts = data['年份'].value_counts().sort_index()
        plt.figure(figsize=(12, 10))
        sns.barplot(x=year_counts.index, y=year_counts.values)
        plt.title('每年发布的游戏数量')
        plt.xlabel('年份')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        plt.savefig("每年发布的游戏数量.svg")
        plt.show()
    def univariate_rating_analysis(self):#好评率的单变量分析
        data = self.steam_data.copy()
        data['好评率'] = pd.to_numeric(
            data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        data = data.dropna(subset=['好评率'])
        print("好评率的描述性统计：")
        print(data['好评率'].describe())
        fliter_data = data[data['好评率']>0]
        plt.figure(figsize=(10, 6))
        sns.histplot(fliter_data['好评率'], kde=True, bins=20)
        plt.title('游戏好评率分布')
        plt.xlabel('好评率 (%)')
        plt.ylabel('频数')
        plt.savefig("好评率分布直方图.svg")
        plt.show()
    '''
        多变量分析
    '''
    
            
    def price_and_goods(self):
        '''
            分析价格与好评率的关系，并加入价格弹性分析
        '''
        data = self.steam_data.copy()
        data['好评率'] = pd.to_numeric(
            data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        data = data.dropna(subset=['价格', '好评率'])

        # 价格分箱（每50元一个区间）
        max_price = int(data['价格'].max()) + 50
        bins = list(range(0, max_price, 50))
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        data['价格区间'] = pd.cut(data['价格'], bins=bins, labels=labels)

        # 按价格区间分组计算平均价格与平均好评率
        grouped = data.groupby('价格区间').agg(
            平均价格=('价格', 'mean'),
            平均好评率=('好评率', 'mean')
        ).reset_index()

        print("各价格区间的平均价格与平均好评率：\n", grouped)

        # 计算价格弹性的辅助列
        grouped['上一区间价格'] = grouped['平均价格'].shift(1)
        grouped['上一区间好评率'] = grouped['平均好评率'].shift(1)

        # 百分比变化
        grouped['价格变化百分比'] = (grouped['平均价格'] - grouped['上一区间价格']) / grouped['上一区间价格']
        grouped['好评率变化百分比'] = (grouped['平均好评率'] - grouped['上一区间好评率']) / grouped['上一区间好评率']

        # 价格弹性 = 好评率变化百分比 / 价格变化百分比
        grouped['价格弹性'] = grouped['好评率变化百分比'] / grouped['价格变化百分比']

        # 删除无用的中间列
        grouped.drop(columns=['上一区间价格', '上一区间好评率'], inplace=True)

        # 打印结果
        print("\n【价格弹性分析结果】")
        print(grouped[['价格区间', '平均价格', '平均好评率', '价格变化百分比', '好评率变化百分比', '价格弹性']].to_string(index=False))

        # 可视化价格弹性
        plt.figure(figsize=(12, 6))
        sns.barplot(x='价格区间', y='价格弹性', data=grouped, palette="coolwarm")
        plt.axhline(0, color='black', linestyle='--')
        plt.title('价格弹性分析（价格对好评率的影响）')
        plt.xlabel('价格区间 (元)')
        plt.ylabel('价格弹性系数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("价格弹性分析.svg")
        plt.show()
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='平均价格', y='平均好评率', data=grouped)
        plt.title('价格与好评率的关系')
        plt.xlabel('平均价格 (元)')
        plt.ylabel('平均好评率 (%)')
        plt.grid(True)
        plt.savefig("./pics/价格与好评率关系散点图.svg")
        plt.show()

        # 新增：计算价格与好评率的相关系数
        correlation = grouped['平均价格'].corr(grouped['平均好评率'])
        print(f"价格与好评率的相关系数: {correlation:.2f}")
        
    def analyze_tag_rating(self): #游戏标签与好评率关系
        data = self.steam_data.copy()
        data['好评率'] = pd.to_numeric(data['好评率'].str.replace('%', '', regex=False), errors='coerce')
        all_tags = []
        for _, row in data.iterrows():
            tags = row['标签'].split('\n') if isinstance(row['标签'], str) else []
            for tag in tags:
                all_tags.append({'tag': tag, '好评率': row['好评率']})
        tag_df = pd.DataFrame(all_tags)
        avg_ratings = tag_df.groupby('tag')['好评率'].mean().sort_values(ascending=False)
        print(avg_ratings.head(20))
        top_tags = avg_ratings.head(20)
        plt.figure(figsize=(14, 8))
        sns.barplot(x=top_tags.values, y=top_tags.index, palette="viridis")
        plt.title("Top 20 标签对应的平均好评率", fontsize=16)
        plt.xlabel("平均好评率 (%)", fontsize=14)
        plt.ylabel("标签", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig("标签与平均好评率关系_top20.svg")
        plt.show()
        print("各标签的平均好评率（前20）：\n", top_tags)
        
    def predict_future_game_price(self):
        data = self.steam_data.copy()
        # 特征工程
        data['年份'] = data['发行日期'].dt.year
        data['好评率数值'] = pd.to_numeric(data['好评率'].str.replace('%', '', regex=False), errors='coerce')
        # 标签 one-hot 编码
        top_tags = ['动作', '冒险', '独立', '策略', '多人']
        for tag in top_tags:
            data[f'标签_{tag}'] = data['标签'].apply(lambda x: 1 if isinstance(x, str) and tag in x else 0)
        # 准备特征与目标变量
        features = ['年份', '好评率数值'] + [f'标签_{tag}' for tag in top_tags]
        X = data[features]
        y = data['价格']
        X = X.fillna(0)
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.evaluate_model(model, X, y, model_name="价格预测模型") #模型评估
        # 构造未来预测数据
        future_data_list = []
        for tag in top_tags:
            row = {
                '年份': 2026,
                '好评率数值': 85,
            }
            for t in top_tags:
                row[f'标签_{t}'] = 1 if t == tag else 0
            future_data_list.append(row)

        future_data = pd.DataFrame(future_data_list)
        predicted_prices = model.predict(future_data)
        # 展示预测结果
        result_df = future_data[['年份', '好评率数值']].copy()
        result_df['预测价格'] = predicted_prices
        for tag in top_tags:
            result_df[tag] = future_data[f'标签_{tag}']
        result_df = result_df.drop('好评率数值',axis='columns')
        print("2026年各类型游戏价格预测：\n",result_df)  
    def predict_future_game_rating(self):
        """
        使用随机森林回归模型预测 2026 年不同类型游戏的平均好评率
        """
        data = self.steam_data.copy()

        # 特征工程
        data['年份'] = data['发行日期'].dt.year
        data['好评率数值'] = pd.to_numeric(
            data['好评率'].str.replace('%', '', regex=False),
            errors='coerce'
        )
        top_tags = ['动作', '冒险', '独立', '策略', '多人']
        for tag in top_tags:
            data[f'标签_{tag}'] = data['标签'].apply(lambda x: 1 if isinstance(x, str) and tag in x else 0)
        features = ['年份', '价格'] + [f'标签_{tag}' for tag in top_tags]
        X = data[features]
        y = data['好评率数值']
        valid_indices = (y > 0) & y.notnull()
        X = X[valid_indices]
        y = y[valid_indices]

        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        #  模型评估
        self.evaluate_model(model, X, y, model_name="好评率预测模型")

        # 构造 2026 年的预测样本数据，每种类型单独预测
        future_data_list = []
        for tag in top_tags:
            row = {
                '年份': 2026,
                '价格': 100,
            }
            for t in top_tags:
                row[f'标签_{t}'] = 1 if t == tag else 0
            future_data_list.append(row)

        future_data = pd.DataFrame(future_data_list)
        predicted_ratings = model.predict(future_data)

        # 输出结果
        result_df = future_data[['年份', '价格']].copy()
        result_df['预测好评率'] = predicted_ratings
        for tag in top_tags:
            result_df[tag] = future_data[f'标签_{tag}']
        result_df = result_df.drop(['价格', '年份'], axis='columns')
        result_df['预测好评率'] = result_df['预测好评率'].apply(lambda x: f'{x:.2f}%')

        print("2026年各类型游戏平均好评率预测：\n", result_df)
    def print_data(self):
        print(self.steam_data)
        
    def evaluate_model(self, model, X, y, model_name="模型"):
        """
        评估模型性能并可视化预测 vs 实际图
        :param model: 训练好的模型
        :param X: 特征数据
        :param y: 目标变量
        :param model_name: 模型名称（用于打印）
        """


        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1. 性能指标
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        mae_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        rmse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')

        print(f"\n【{model_name} 评估结果】")
        print(f'R²: {scores.mean():.2f} ± {scores.std():.2f}')
        print(f'MAE: {-mae_scores.mean():.2f} ± {-mae_scores.std():.2f}')
        print(f'RMSE: {-rmse_scores.mean():.2f} ± {-rmse_scores.std():.2f}')

        # 2. 可视化预测 vs 实际
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想线
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} 预测值 vs 实际值')
        plt.tight_layout()
        plt.savefig(f"{model_name}_预测效果.svg")
        plt.show()

        # 3. 特征重要性分析
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            print("\n特征重要性：")
            print(feat_imp)
            
def main():
    path = '2.xlsx'
    qte = steam_ana(path)
    qte.lose_solve()  # 处理缺失值
    #qte.draw_datas()  # 绘制各类数据图表
    #qte.ana_goods_rate(phrase="好评如潮")  # 分析好评率挡位占比
    #qte.univariate_year_analysis()  # 发行年份的单变量分析
    #qte.univariate_rating_analysis()  # 好评率的单变量分析
    qte.price_and_goods()  # 分析价格与好评率的关系
    #qte.analyze_tag_rating()  # 游戏标签与好评率关系分析
    #qte.predict_future_game_price()  # 预测未来游戏价格
    #qte.predict_future_game_rating() #预测未来游戏价格
    return 0

if __name__ == "__main__":
    main()