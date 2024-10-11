import numpy as np
import pandas as pd
import streamlit as st
import graph_function


# CSVファイルの読み込み関数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df["Flop action 1"]=='bet']
    return df

#ブラフEV計算関数
def bluff_ev(row):
    if row["Flop action 2"] == "fold":
        return row["Pot(BB)"]
    return (row["Pot(BB)"]*row["Flop size 1"]) * -0.01

#ファイルパスの指定
mda_path = 'https://drive.google.com/uc?export=download&id=1ojUmB6fs4JYyRjD3f2qAEaPqKzW4wdX7'
gto_path = 'https://drive.google.com/uc?export=download&id=1FA4hfKhQB0XgVJ-lNq3NRpiHlN2LMPek'

#ファイルの読み込み
df_mda = load_data(mda_path)
df_gto = load_data(gto_path)

#列の追加
num_categories = ["2-9","10-Q","K","A"]
df_mda["Flop_rank"] = pd.cut(df_mda["Flop_high"],
                      bins=[1,9,12,13,15],
                      labels=num_categories,
                      ordered=True)
df_gto["Flop_rank"] = pd.cut(df_gto["Flop_high"],
                      bins=[1,9,12,13,15],
                      labels=num_categories,
                      ordered=True)
player_categories = ["0-1k","1k-10k","10k-100k","over 100k"]
df_mda["player_rank"] = pd.cut(df_mda["IP Hands Played"],
                        bins=[0,1000,10000,100000,np.inf],
                        labels=player_categories,
                        ordered=True)
size_categories = ["min-33%","33%-66%","66%-100%","potover"]
df_mda["Flop_bet_size"] = pd.cut(df_mda["Flop size 1"],
                             bins=[0,33,66,100,np.inf],
                             labels=size_categories,
                             ordered=True)
df_gto["Flop_bet_size"] = df_gto["Flop size 1"].round().astype(int)

#サイドバー設定
feature_col_mda = ["player_rank","Flop_rank","Flop_class"]
feature_col_gto = ["Flop_rank","Flop_class"]

with st.sidebar.form(key="sidebar_form"):
    with st.expander("ヒートマップの設定"):
        pivot_x = st.selectbox("ヒートマップのx軸",feature_col_mda,index=0)
        pivot_y = st.selectbox("ヒートマップのy軸",feature_col_mda,index=1)
        pivot_z = st.selectbox("ヒートマップのz軸",["Bluff EV","Raise%"])
    with st.expander("ボード設定(弾性と役分布用)"):
        player_rank = st.selectbox("プレイヤーランクの選択",["All","0-1k","1k-10k","10k-100k","over 100k"])
        flop_high  = st.selectbox("ハイカードの選択",["All","2-9","10-Q","K","A"])
        flop_class  = st.selectbox("ボード種類の選択",["All","Paired","Monotone","Rainbow","Twotone"])
    with st.expander("役分布の設定"):
        action = st.selectbox("アクションを選択",["call","raise"])
        mda_flop_size = st.selectbox("MDAの前ノードのbetサイズを選択",["All"]+list(df_mda["Flop_bet_size"].unique()))
        gto_flop_size = st.selectbox("GTOの前ノードのbetサイズを選択",["All"]+list(df_gto["Flop_bet_size"].unique()))
    submit_button = st.form_submit_button(label='更新')

#pivot目的変数の設定
if pivot_z == "Bluff EV":
    df_mda["target"] = df_mda.apply(bluff_ev, axis=1)
    df_gto["target"] = df_gto.apply(bluff_ev, axis=1)
else:
    df_mda["target"] = (df_mda["Flop action 2"] == 'raise').astype(int)
    df_gto["target"] = (df_gto["Flop action 2"] == 'raise').astype(int)

st.header("betted node テスト")
st.text("相手がbetもしくはraiseした後のノードを対象としています")
st.text("BBvsBTN3betでBBのCBを受けたBTNのアクションをサンプルデータとしています")

#ヒートマップの表示
st.subheader("ピボットヒートマップ")
st.text("ピボットのセルの数値はBluff EVかRaise頻度を選択します")
st.text("Bluff EVはEQ0%のハンドのbetEV(BB)を表しています")
fig1 = graph_function.plot_heatmap(df_mda,df_gto,pivot_x,pivot_y,"target")
st.plotly_chart(fig1, use_container_width=True)

#表データの表示
st.subheader("表分析")
st.text("Targetの数値はピボットと同様の設定です")
dfg = graph_function.plot_table(df_mda,df_gto,feature_col_mda,feature_col_gto,"target")
st.dataframe(dfg)

#データの絞り込み
df_mda_filtered = df_mda.copy()
df_gto_filtered = df_gto.copy()
if player_rank != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["player_rank"]==player_rank]
if flop_high != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop_rank"]==flop_high]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop_rank"]==flop_high]
if flop_class != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop_class"]==flop_class]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop_class"]==flop_class]

#弾性グラフの表示
fig2 = graph_function.plot_elasticity(df_mda_filtered,"Flop size 1","Flop action 2", df_gto_filtered)
st.plotly_chart(fig2, use_container_width=True)

#アクションの絞り込み
if action == "call":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop action 2"]=="call"]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop action 2"]=="call"]
else:
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop action 2"]=="raise"]
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop action 2"]=="raise"]

#サイズの絞り込み
if mda_flop_size != "All":
    df_mda_filtered = df_mda_filtered[df_mda_filtered["Flop_bet_size"]==mda_flop_size]
if gto_flop_size != "All":
    df_gto_filtered = df_gto_filtered[df_gto_filtered["Flop_bet_size"]==gto_flop_size]

#SD係数の取得
df_coef,dic_coef = graph_function.get_flop_sd_bias(df_gto_filtered,"IP")

#バイアスあり役分布の表示
st.subheader("アクション毎の役構成(SDバイアスあり)")
df_gto_filtered_bias = df_gto_filtered[df_gto_filtered["IP_sd"]==1]
fig3 = graph_function.plot_range(df_mda_filtered,df_gto_filtered_bias,"IP_Flop_hand_rank")
st.plotly_chart(fig3, use_container_width=True)

#バイアス除去レンジとSD係数の表示
st.subheader("アクション毎の役構成(SDバイアス除去)")
fig4 = graph_function.plot_range_nobias(df_mda_filtered,df_gto_filtered,"IP_Flop_hand_rank",dic_coef)
st.plotly_chart(fig4, use_container_width=True)