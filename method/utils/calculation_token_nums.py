import ast

import pandas as pd


def calculate_token_nums():
    try:
        df = pd.read_csv('method\\store\\token\\all_token.csv',
                         header=None,
                         sep='|',  # 使用一个文件中不存在的字符作为分隔符
                         quotechar="'",
                         engine='python'
                         )

        def extract_tokens(row_str):
            if pd.isna(row_str) or not isinstance(row_str, str):
                return 0
            try:
                data_dict = ast.literal_eval(row_str)
                return data_dict.get('total_tokens', 0)
            except (ValueError, SyntaxError):
                return 0

        # 3. 关键改动：通过索引 0 直接访问第一列
        df['total_tokens'] = df.iloc[:, 0].apply(extract_tokens)

        # 4. 计算总和
        total_sum = df['total_tokens'].sum()

        return int(total_sum)
    except Exception as e:
        return 0


def calculate_token_nums_simplt():
    try:
        df = pd.read_csv('method\\store\\token\\all_token.csv',
                         header=None,
                         sep='|',  # 使用一个文件中不存在的字符作为分隔符
                         quotechar="'",
                         engine='python'
                         )
        print(df.values.sum())
        return int(df.values.sum())
    except Exception as e:
        return 0


def clear_token_csv_file():
    with open('method\\store\\token\\all_token.csv', 'w') as f:
        f.write('')


if __name__ == '__main__':
    calculate_token_nums_simplt()