from io import BytesIO
from ALL_Class.Helper import *
from ALL_Class.op import * 
from ALL_Class.Module import * 
class TxFetcher:
    cache = {}
    @classmethod
    def get_url(cls, testnet = False):
        if testnet :
            return "https://blockstream.info/testnet/api"
        else:
            return "https://blockstream.info/api"
    
    @classmethod
    def fetch(cls , tx_id, testnet = False , fresh = False):
        from ALL_Class.Transaction import Tx
        if fresh or (tx_id not in cls.cache):
            url = "{}/tx/{}/hex".format(cls.get_url(testnet), tx_id)
            response = requests.get(url)
            try:
                raw = bytes.fromhex(response.text.strip())
            except:
                raise ValueError('unexpected response: {}'.format(response.text))
            
        if raw[4] == 0:
            raw = raw[:4] + raw[6:]
            tx = Tx.parse(BytesIO(raw), testnet = testnet)
            tx.locktime = little_endian_to_int(raw[-4:])
        else:
            tx = Tx.parse(BytesIO(raw), testnet = testnet)
        cls.cache[tx_id] = tx
        return cls.cache[tx_id]