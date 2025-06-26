import sys
import os
import datetime
import psutil
import time
from blockchain_parser.blockchain import Blockchain
from typing import Dict, Any

COINBASE = "0000000000000000000000000000000000000000000000000000000000000000"

def isvalid(ou) -> bool:
    """檢查output是否為有效地址與金額"""
    # 假設 ou.addresses[0] 總是存在，並且不是未知地址
    return (len(ou.addresses) > 0 and not ou.is_unknown() and ou.value > 0)

# 移除了 index_path 參數，因為 get_unordered_blocks() 不需要它
def parsing(
    blockchain_path: str,
    end: datetime.datetime,
    out_dir: str
) -> None:
    """從區塊資料庫解析交易鏈，輸出關聯交易到檔案"""
    tran_path = os.path.join(out_dir, 'tran.txt')
    log_path = os.path.join(out_dir, 'parsing_log.txt')
    trans: Dict[str, Dict[int, Dict[str, Any]]] = {}

    blk_count = 0
    t0 = time.time()

    with open(tran_path, 'w') as tran_file, open(log_path, 'a') as logging:
        # 初始化 Blockchain，只傳遞區塊資料的路徑
        print("Initializing Blockchain...") # 新增即時訊息
        blockchain = Blockchain(os.path.expanduser(blockchain_path))
        print("Blockchain initialized. Starting block parsing...") # 新增即時訊息
        
        # *** 關鍵修改：使用 get_unordered_blocks() ***
        # 這個方法會按照檔案系統中找到的順序解析區塊，不需要 LevelDB 索引
        for block in blockchain.get_unordered_blocks(): 
            # 直接使用 block.header.timestamp (假設它已是 datetime 物件)
            # 並且修正 DeprecationWarning，使用 datetime.datetime.fromtimestamp()
            # 但如果它已經是 datetime 物件，就直接用它
            block_timestamp_dt = block.header.timestamp # 假設它已經是 datetime 物件
            
            print(f"Processing block: Height {block.height}, Hash {block.hash[:10]}... Time: {block_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # 直接使用 block_timestamp_dt 進行比較
                if block_timestamp_dt > end:
                    print(f"Reached end date {end.strftime('%Y-%m-%d %H:%M:%S')}. Stopping parsing.") # 新增停止訊息
                    break
                for t in block.transactions:
                    try:
                        outputs = []
                        total_val = 0
                        for ou in t.outputs:
                            if isvalid(ou):
                                outputs.append({'value': ou.value, 'address': ou.addresses[0].address})
                                total_val += ou.value
                        if len(outputs) > 0:
                            trans[t.hash] = dict(zip(range(len(outputs)), outputs))
                        else:
                            continue
                        for ou in t.outputs:
                            if not isvalid(ou):
                                continue
                            # 檢查 total_val，避免除以零
                            if total_val == 0:
                                logging.write(f"Warning: Transaction {t.hash} has total_val of 0, skipping fraction calculation for an output.\n")
                                continue

                            fraction = ou.value / total_val
                            for i in t.inputs:
                                if i.transaction_hash != COINBASE:
                                    prev_hash = i.transaction_hash
                                    index = i.transaction_index
                                    if prev_hash not in trans or index not in trans[prev_hash]:
                                        # 記錄下找不到前一個交易輸出的情況
                                        logging.write(f"Warning: Previous output {prev_hash}:{index} not found for input in transaction {t.hash}. Possibly coinbase or unspent.\n")
                                        continue
                                    prev_ou = trans[prev_hash][index]
                                    prev_addr = prev_ou['address']
                                    tran_file.write(
                                        f"{prev_addr},{ou.addresses[0].address},{t.hash},"
                                        f"{round(prev_ou['value'] * fraction, 2)},"
                                        f"{block_timestamp_dt.timestamp()},{block.height}\n" # 使用 datetime 物件的 timestamp() 方法
                                    )
                                    del trans[prev_hash][index]
                                    if len(trans[prev_hash]) == 0:
                                        del trans[prev_hash]
                    except Exception as e:
                        logging.write(f"Transaction error for TX {t.hash}: {str(e)}\n")
                        continue
                blk_count += 1
                if blk_count % 100 == 0:
                    if blk_count > 100:
                        logging.write('===================================================\n')
                    logging.write(f'Blocks: {blk_count}\n')
                    logging.write(f'Time: {time.time() - t0}\n')
                    logging.write(f'CPU_percent: {psutil.cpu_percent()}\n')
                    logging.write(f'Memory: {psutil.virtual_memory()}\n')
            except Exception as e:
                logging.write(f"Block error for block {block.hash} at height {block.height}: {str(e)}\n")
                continue
            del block
        logging.write(f'Total_time: {time.time() - t0}\n')
    print('Done')

def main():
    if len(sys.argv) < 4:
        print("Usage: python parsing.py <blockchain_path> <YYYY-MM-DD> <out_dir>")
        sys.exit(1)

    blockchain_path = sys.argv[1]
    end_str = sys.argv[2]
    out_dir = sys.argv[3]

    # 日期格式檢查
    try:
        end = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        print("Error: end date must be inYYYY-MM-DD format.")
        sys.exit(1)

    # 處理輸出目錄
    if not (out_dir.startswith('./') or out_dir.startswith('/')):
        out_dir = './' + out_dir
    if not out_dir.endswith('/'):
        out_dir += '/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 由於使用 get_unordered_blocks()，不再需要 LevelDB 索引路徑
    # 移除 index_path_for_parser 的定義和 os.makedirs
    
    # 調用 parsing 函數時，不再傳遞 index_path 參數
    parsing(blockchain_path, end, out_dir) 

if __name__ == '__main__':
    main()
