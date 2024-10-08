import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np


hand_rank = ["4C","FH", "Fl", "St", "3C", "2P", "OP", "TP(GK)", "TP(LK)", "MP", "LP", "No"]
color_list = ['#0000ff', '#0044ff', '#0088ff', '#00ccff', '#00ffcc', '#00ff88', '#00ff00', '#88ff00', '#ffcc00', '#ff8800', '#ff4400', '#ff0000']
color_dic = dict(zip(hand_rank, color_list))


#ヒートマップの作成(df_a=MDA,df_b=GTO)
def plot_heatmap(df_a, df_b, col_x, col_y, col_z):
    # pivot_aの作成（MDA値）
    pivot_a = pd.pivot_table(df_a, values=col_z, index=col_y, columns=col_x, aggfunc='mean')
    
    # df_bの処理（GTO値）
    if col_x in df_b.columns and col_y in df_b.columns:
        pivot_b = pd.pivot_table(df_b, values=col_z, index=col_y, columns=col_x, aggfunc='mean')
    elif col_x in df_b.columns:
        pivot_b = df_b.groupby(col_x)[col_z].mean().to_frame().T
        pivot_b = pd.DataFrame(np.tile(pivot_b.values, (len(pivot_a.index), 1)), 
                               index=pivot_a.index, columns=pivot_a.columns)
    elif col_y in df_b.columns:
        pivot_b = df_b.groupby(col_y)[col_z].mean()
        pivot_b = pd.DataFrame(np.tile(pivot_b.values, (len(pivot_a.columns), 1)).T, 
                               index=pivot_a.index, columns=pivot_a.columns)
    else:
        mean_value = df_b[col_z].mean()
        pivot_b = pd.DataFrame(mean_value, index=pivot_a.index, columns=pivot_a.columns)
    
    # ヒートマップの作成
    fig = go.Figure(data=go.Heatmap(
        z=pivot_a.values,
        x=pivot_a.columns,
        y=pivot_a.index,
        hoverongaps=False,
        colorscale='Viridis'
    ))
    
    # テキストの追加（MDAとGTO両方の値を表示）
    text_matrix = [[f'MDA: {a:.2f}<br>GTO: {b:.2f}' for a, b in zip(row_a, row_b)] 
                   for row_a, row_b in zip(pivot_a.values, pivot_b.values)]
    
    fig.update_traces(text=text_matrix, 
                      texttemplate="%{text}", 
                      textfont={"size":10})
    
    # レイアウトの設定
    fig.update_layout(
        xaxis_title=col_x,
        yaxis_title=col_y
    )
    
    return fig


#表データの作成
def plot_table(df_mda, df_gto, cols_mda, cols_gto,target):
    dfg_mda = df_mda.groupby(cols_mda,observed=True).agg({target:['mean', 'count']}).reset_index()
    dfg_mda.columns = [*cols_mda, "Target_MDA", "Data count"]
    dfg_mda["Target_MDA"] = dfg_mda["Target_MDA"].round(2)

    dfg_gto = df_gto.groupby(cols_gto,observed=True).agg({target:['mean']}).reset_index()
    dfg_gto.columns = [*cols_gto, "Target_GTO"]
    dfg_gto["Target_GTO"] = dfg_gto["Target_GTO"].round(2)

    dfg = dfg_mda.merge(dfg_gto,on=cols_gto,how="left", suffixes=('', '_GTO'))
    dfg["GTO - MDA"] = dfg["Target_GTO"] - dfg["Target_MDA"]

    col_list = [*cols_mda, "Target_MDA", "Target_GTO", "Data count","GTO - MDA"]
    return dfg[col_list]


#弾性グラフの作成
def plot_elasticity(df, x_col, y_col, df_gto=None, bin_size=10, bin_threshold=20):
    # ビンの作成
    bins = np.arange(0, 201, bin_size)
    df = df.copy()
    df['bin'] = pd.cut(df[x_col], bins=bins)

    # ビン毎の確率計算
    grouped = df.groupby('bin', observed=True)
    totals = grouped.size()
    cr_counts = grouped[y_col].apply(lambda x: ((x == 'call') | (x == 'raise')).sum())
    r_counts = grouped[y_col].apply(lambda x: (x == 'raise').sum())

    p_cr = (cr_counts / totals).reset_index()
    p_cr.columns = ['bin', 'p_cr']

    p_r = (r_counts / totals).reset_index()
    p_r.columns = ['bin', 'p_r']

    # ビンの中央値計算
    p_cr['x'] = p_cr['bin'].apply(lambda x: x.mid)
    p_r['x'] = p_r['bin'].apply(lambda x: x.mid)

    # ビンのサンプル数計算
    bin_counts = totals.reset_index()
    bin_counts.columns = ['bin', 'count']
    bin_counts['x'] = bin_counts['bin'].apply(lambda x: x.mid)

    # 閾値未満のビンを除外
    valid_bins = bin_counts[bin_counts['count'] >= bin_threshold]['bin']
    p_cr = p_cr[p_cr['bin'].isin(valid_bins)]
    p_r = p_r[p_r['bin'].isin(valid_bins)]
    bin_counts = bin_counts[bin_counts['bin'].isin(valid_bins)]

    # サブプロット作成
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])

    # グラフ追加 (call or raise)
    fig.add_trace(go.Scatter(x=p_cr['x'], y=p_cr['p_cr'], 
                             mode='lines+markers', name='Call+Raise freq'), row=1, col=1)

    # グラフ追加 (raise only)
    fig.add_trace(go.Scatter(x=p_r['x'], y=p_r['p_r'], 
                             mode='lines+markers', name='Raise freq'), row=1, col=1)

    # MDF曲線追加
    x_range = np.linspace(0, 200, 1000)
    y_curve = 1 / (1 + x_range / 100)
    fig.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', name="MDF"),
                  row=1, col=1)

    # GTO確率計算と追加
    if not df_gto.empty:
        df_gto = df_gto.copy()
        df_gto['bin'] = pd.cut(df_gto[x_col], bins=bins)
        gto_grouped = df_gto.groupby('bin', observed=True)
        gto_totals = gto_grouped.size()
        gto_cr_counts = gto_grouped[y_col].apply(lambda x: ((x == 'call') | (x == 'raise')).sum())
        gto_r_counts = gto_grouped[y_col].apply(lambda x: (x == 'raise').sum())
        
        gto_p_cr = (gto_cr_counts / gto_totals).reset_index()
        gto_p_cr.columns = ['bin', 'p_cr']
        gto_p_cr['x'] = gto_p_cr['bin'].apply(lambda x: x.mid)
        gto_p_cr = gto_p_cr[gto_p_cr['bin'].isin(valid_bins)]
        
        gto_p_r = (gto_r_counts / gto_totals).reset_index()
        gto_p_r.columns = ['bin', 'p_r']
        gto_p_r['x'] = gto_p_r['bin'].apply(lambda x: x.mid)
        gto_p_r = gto_p_r[gto_p_r['bin'].isin(valid_bins)]
        
        fig.add_trace(go.Scatter(x=gto_p_cr['x'], y=gto_p_cr['p_cr'], 
                                mode='lines+markers', name='GTO Call+Raise freq'), row=1, col=1)
        fig.add_trace(go.Scatter(x=gto_p_r['x'], y=gto_p_r['p_r'], 
                                mode='lines+markers', name='GTO Raise freq'), row=1, col=1)

    # BluffEVグラフ追加
    bluff_ev = [(100*(1-y)) - (x*y) for x, y in zip(p_cr['x'], p_cr['p_cr'])]
    fig.add_trace(go.Bar(x=bin_counts['x'], y=bluff_ev, name="Bluff EV"),
                  row=2, col=1)

    # レイアウト設定
    fig.update_layout(
        yaxis_title='Frequency',
        yaxis_tickformat='.0%',
        xaxis2_title='Bet Size (%)',
        yaxis2_title='Bluff EV',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=800
    )

    # Y軸の範囲を0-1に設定（上部のグラフ）
    fig.update_yaxes(range=[0, 1], row=1, col=1)

    return fig


#SDバイアスありレンジ構成の可視化(df_gtoにはバイアスを掛けものを渡す)
def plot_range(df_mda, df_gto, col, font_size = 12, color_map = color_dic, category_order = hand_rank):
    # 両データフレームの指定された列のユニークな値の割合を計算
    value_counts_mda = df_mda[col].value_counts(normalize=True)
    value_counts_gto = df_gto[col].value_counts(normalize=True)
    
    # 指定された順序に基づいてvalue_countsをソート
    value_counts_mda = value_counts_mda.reindex(category_order).fillna(0)
    value_counts_gto = value_counts_gto.reindex(category_order).fillna(0)
    
    # データをプロット用に整形
    labels = category_order
    values_mda = value_counts_mda.values
    values_gto = value_counts_gto.values
    
    data = []
    
    # MDAのグラフを作成
    for label, value in zip(reversed(labels), reversed(values_mda)):
        color = color_map[label]
        data.append(go.Bar(
            y=['MDA'],
            x=[value],
            name=label,
            marker=dict(color=color),
            text=[f'{label}: {value:.1%}'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False
        ))
    
    # GTOのグラフを作成
    for label, value in zip(reversed(labels), reversed(values_gto)):
        color = color_map[label]
        data.append(go.Bar(
            y=['GTO'],
            x=[value],
            name=label,
            marker=dict(color=color),
            text=[f'{label}: {value:.1%}'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False  # MDAのグラフで既に凡例が表示されているため
        ))
    
    # レイアウトの設定
    layout = go.Layout(
        barmode='stack',
        xaxis=dict(
            title='Frequency',
            range=[0, 1],
            tickformat=',.0%',
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=['GTO', 'MDA']  # GTOを下に、MDAを上に配置
        ),
        height=600,  # グラフの高さを調整（2つのグラフを表示するため）
        margin=dict(l=150)
    )
    
    # 図の作成
    fig = go.Figure(data=data, layout=layout)
    
    return fig


#SDバイアス係数の取得
def get_flop_sd_bias(df,position):
    if position == "OOP":
        df_bias = df[df["OOP_sd"] == 1]
        col = "OOP_Flop_hand_rank"
    else:
        df_bias = df[df["IP_sd"] == 1]
        col = "IP_Flop_hand_rank"
    
    count_a = df[col].value_counts().reset_index()
    count_b = df_bias[col].value_counts().reset_index()
    
    # カラム名を変更
    count_a.columns = [col, 'count_a']
    count_b.columns = [col, 'count_b']
    
    # 2つのデータフレームをマージ
    merged = pd.merge(count_a, count_b, on=col, how='outer').fillna(0)
    
    # 比率を計算
    merged['coefficient'] = merged['count_a'] / merged['count_b']
    merged['coefficient'] = merged['coefficient'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # 結果のデータフレームを作成
    result = merged[[col, 'coefficient']].set_index(col)
    dic = result['coefficient'].to_dict()
    
    return result,dic


#SDバイアスなしレンジの可視化
def plot_range_nobias(df_mda, df_gto, col, coefficient_map, font_size=12, color_map=color_dic, category_order=hand_rank):
    # df_mdaの処理: カテゴリ列の値毎のカウントを集計し、係数を適用して再集計
    value_counts_mda = df_mda[col].value_counts()
    adjusted_counts_mda = value_counts_mda * value_counts_mda.index.map(coefficient_map.get)
    value_counts_mda = adjusted_counts_mda / adjusted_counts_mda.sum()

    # df_gtoの処理: 変更なし
    value_counts_gto = df_gto[col].value_counts(normalize=True)
    
    # 指定された順序に基づいてvalue_countsをソート
    value_counts_mda = value_counts_mda.reindex(category_order).fillna(0)
    value_counts_gto = value_counts_gto.reindex(category_order).fillna(0)
    
    # データをプロット用に整形
    labels = category_order
    values_mda = value_counts_mda.values.tolist()
    values_gto = value_counts_gto.values.tolist()
    
    data = []
    
    # MDAのグラフを作成
    for label, value in zip(reversed(labels), reversed(values_mda)):
        color = color_map[label]
        data.append(go.Bar(
            y=['MDA'],
            x=[value],
            name=label,
            marker=dict(color=color),
            text=[f'{label}: {value:.1%}'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False
        ))
    
    # GTOのグラフを作成
    for label, value in zip(reversed(labels), reversed(values_gto)):
        color = color_map[label]
        data.append(go.Bar(
            y=['GTO'],
            x=[value],
            name=label,
            marker=dict(color=color),
            text=[f'{label}: {value:.1%}'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False  # MDAのグラフで既に凡例が表示されているため
        ))
    
    # レイアウトの設定
    layout = go.Layout(
        barmode='stack',
        xaxis=dict(
            title='Frequency',
            range=[0, 1],
            tickformat=',.0%',
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=['GTO', 'MDA']  # GTOを下に、MDAを上に配置
        ),
        height=600,  # グラフの高さを調整（2つのグラフを表示するため）
        margin=dict(l=150)
    )
    
    # 図の作成
    fig = go.Figure(data=data, layout=layout)
    
    return fig


#アクション構成の可視化
def plot_action(df_mda, df_gto, col_mda, col_gto, mda_order, font_size=12):
    # 両データフレームの指定された列のユニークな値の割合を計算
    value_counts_mda = df_mda[col_mda].value_counts(normalize=True)
    value_counts_gto = df_gto[col_gto].value_counts(normalize=True)

    # 指定された順序に基づいてvalue_countsをソート
    mda_labels = ["check", *mda_order]
    value_counts_mda = value_counts_mda.reindex(mda_labels).fillna(0)

    # GTOラベルを"check"を先頭にしてラベルの値で並べ替え
    gto_labels = ["check"] + sorted([label for label in value_counts_gto.index if label != "check"])

    # データをプロット用に整形
    values_mda = value_counts_mda.values
    values_gto = value_counts_gto.reindex(gto_labels).fillna(0).values

    # グラデーションカラーリストの作成（黄から赤へのグラデーション）
    def create_gradient_colors(num_colors):
        return [f'rgb({int(r)}, {int(g)}, 0)' for r, g in zip(np.linspace(255, 255, num_colors), np.linspace(255, 0, num_colors))]

    mda_colors = ["rgb(0, 255, 0)" if label == "check" else color for label, color in zip(mda_labels, create_gradient_colors(len(values_mda)))]
    gto_colors = ["rgb(0, 255, 0)" if label == "check" else color for label, color in zip(gto_labels, create_gradient_colors(len(values_gto)))]

    data = []

    # MDAのグラフを作成（積み上げ順を逆に）
    for label, value, color in zip(mda_labels, values_mda, mda_colors):
        data.append(go.Bar(
            y=['MDA'],
            x=[value],
            name=f'{label}: {int(value * 100)}%',
            marker=dict(color=color),
            text=[f'{label}: {int(value * 100)}%'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False
        ))

    # GTOのグラフを作成（"check"を先頭に、ラベルの値で積み上げ）
    for label, value, color in zip(gto_labels, values_gto, gto_colors):
        data.append(go.Bar(
            y=['GTO'],
            x=[value],
            name=f'{label}: {int(value * 100)}%',
            marker=dict(color=color),
            text=[f'{label}%: {int(value * 100)}%'],
            textposition='inside',
            orientation='h',
            textfont=dict(size=font_size),
            showlegend=False
        ))

    # レイアウトの設定
    layout = go.Layout(
        barmode='stack',
        xaxis=dict(
            title='Frequency',
            range=[0, 1],
            tickformat=',.0%',
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=['GTO', 'MDA']  # GTOを下に、MDAを上に配置
        ),
        height=600,  # グラフの高さを調整（2つのグラフを表示するため）
        margin=dict(l=150)
    )

    # 図の作成
    fig = go.Figure(data=data, layout=layout)

    return fig