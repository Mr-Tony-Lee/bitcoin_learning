import sys
import os
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, List

def load_address_user_maps(a2u_path: str, u2a_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    載入地址與用戶的映射檔案。
    """
    a2u: Dict[str, str] = {}
    u2a: Dict[str, List[str]] = {}
    with open(a2u_path, 'r') as a2u_file:
        for line in a2u_file:
            if ':' in line:
                addr, uid = line.strip().split(':', 1)
                a2u[addr] = uid
    with open(u2a_path, 'r') as u2a_file:
        for line in u2a_file:
            if ':' in line:
                uid, addr_str = line.strip().split(':', 1)
                u2a[uid] = addr_str.split(',') if addr_str else []
    return a2u, u2a

def build_user_graph(tran_path: str, a2u: Dict[str, str], output_path: str) -> None:
    """
    將交易資料中的地址轉換為用戶之間的圖，並輸出用戶間的連結。
    """
    df = pd.read_csv(tran_path, header=None)
    # 假設tran.txt格式為：from_addr, to_addr, tx_hash, value, timestamp, block_height
    user_links = set()
    for _, row in df.iterrows():
        from_addr = row[0]
        to_addr = row[1]
        from_uid = a2u.get(str(from_addr), None)
        to_uid = a2u.get(str(to_addr), None)
        if from_uid is not None and to_uid is not None and from_uid != to_uid:
            # 只記錄不同用戶間的連結
            user_links.add((from_uid, to_uid))
    # 輸出 user_links.txt
    with open(output_path, 'w') as f:
        for u, v in user_links:
            f.write(f"{u},{v}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ugraph.py <dir>")
        sys.exit(1)
    _dir = sys.argv[1]
    if not (_dir.startswith('./') or _dir.startswith('/')):
        _dir = './' + _dir
    if not _dir.endswith('/'):
        _dir += '/'
    if not os.path.isdir(_dir):
        print(f"Directory {_dir} does not exist.")
        sys.exit(1)
    tran_path = os.path.join(_dir, 'tran.txt')
    a2u_path = os.path.join(_dir, 'a2u.txt')
    u2a_path = os.path.join(_dir, 'u2a.txt')
    output_path = os.path.join(_dir, 'user_links.txt')
    if not (os.path.isfile(tran_path) and os.path.isfile(a2u_path) and os.path.isfile(u2a_path)):
        print("Required files (tran.txt, a2u.txt, u2a.txt) not found in the specified directory.")
        sys.exit(1)
    a2u, u2a = load_address_user_maps(a2u_path, u2a_path)
    build_user_graph(tran_path, a2u, output_path)
    print(f"user_links.txt has been created at {output_path}")

if __name__ == '__main__':
    main()