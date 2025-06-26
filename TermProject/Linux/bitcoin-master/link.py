"""
    # link.py
    將一份交易資料進行“關聯分析”，自動連結出同一群相關聯的地址，並產生多個中介檔案，方便之後分析比特幣交易網絡中的「群組」或「用戶」
"""
import sys
import os
import time
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, Dict, List

def find_links(tran: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    依據交易資料，產生 links.txt 及 along.txt，並返回其內容的 DataFrame。
    """
    p1 = 'links.txt'
    p2 = 'along.txt'
    if os.path.isfile(p1) and os.path.isfile(p2):
        return pd.read_csv(p1, header=None), pd.read_csv(p2, header=None)
    if tran is None or tran.empty:
        raise ValueError("Input 'tran' DataFrame is required and cannot be empty.")

    prev_tid = ''
    group: List[str] = []
    # 避免 SettingWithCopyWarning，直接複製一份
    tran = pd.concat([tran, tran.iloc[[0]]], ignore_index=True)
    t0 = time.time()
    with open('linking_log.txt', 'a') as logging, open(p1, 'w') as f1, open(p2, 'w') as f2:
        for _, row in tran.iterrows():
            if row[2] == prev_tid:
                group.append(row[0])
            else:
                if len(group) > 1:
                    for i in range(1, len(group)):
                        f1.write(f"{group[i-1]},{group[i]}\n")
                elif len(group) == 1:
                    for a in group:
                        f2.write(f"{a}\n")
                group = [row[0]]
                prev_tid = row[2]
        logging.write(f"Time for linking: {time.time() - t0}\n")
    return pd.read_csv(p1, header=None), pd.read_csv(p2, header=None)

def get_users(
    linked_df: Optional[pd.DataFrame] = None, 
    along_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    將 links.txt 與 along.txt 轉為 user group 映射檔案，並返回 a2u, u2a 字典。
    """
    p1 = 'a2u.txt'
    p2 = 'u2a.txt'
    if os.path.isfile(p1) and os.path.isfile(p2):
        a2u: Dict[str, str] = {}
        u2a: Dict[str, List[str]] = {}
        with open(p1, 'r') as a2u_file, open(p2, 'r') as u2a_file:
            for line in a2u_file:
                if ':' in line:
                    addr, uid = line.strip().split(':', 1)
                    a2u[addr] = uid
            for line in u2a_file:
                if ':' in line:
                    uid, addr_str = line.strip().split(':', 1)
                    u2a[uid] = addr_str.split(',') if addr_str else []
        return a2u, u2a
    if linked_df is None or along_df is None:
        raise ValueError("linked_df and along_df are required when user mapping files do not exist.")

    t0 = time.time()
    with open('linking_log.txt', 'a') as logging, open(p1, 'w') as a2u_file, open(p2, 'w') as u2a_file:
        G = nx.from_pandas_edgelist(linked_df, 0, 1)
        G.add_nodes_from(along_df[0].values)
        groups = nx.connected_components(G)
        uid = 0
        a2u: Dict[str, str] = {}
        u2a: Dict[str, List[str]] = {}
        for grp in groups:
            addr_list = list(grp)
            u2a[str(uid)] = addr_list
            u2a_file.write(f"{uid}:{','.join(addr_list)}\n")
            for a in addr_list:
                a2u[a] = str(uid)
                a2u_file.write(f"{a}:{uid}\n")
            uid += 1
        logging.write(f"Time for parsing: {time.time() - t0}\n")
    return a2u, u2a

def main():
    if len(sys.argv) < 2:
        print("Usage: python link.py <directory>")
        sys.exit(1)
    _dir = sys.argv[1]
    if not (_dir.startswith('./') or _dir.startswith('/')):
        _dir = './' + _dir
    if not os.path.isdir(_dir):
        print(f"Directory {_dir} does not exist.")
        sys.exit(1)
    os.chdir(_dir)
    if not os.path.isfile('tran.txt'):
        print("tran.txt not found in the specified directory.")
        sys.exit(1)
    tran = pd.read_csv('tran.txt', header=None)
    link_df, along_df = find_links(tran)
    get_users(link_df, along_df)

if __name__ == '__main__':
    main()