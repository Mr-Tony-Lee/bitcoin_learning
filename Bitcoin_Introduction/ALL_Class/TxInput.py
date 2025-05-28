from ALL_Class.Helper import *
from ALL_Class.op import * 
from ALL_Class.Module import * 


class TxIn:
    def __init__(self, prev_tx, prev_index, script_sig, sequence):
        from ALL_Class.Script import Script
        self.prev_tx = prev_tx    # Previous transaction ID
        self.prev_index = prev_index    # Output index of the previous transaction
        if script_sig is None:
            self.script_sig = Script()
        else:
            self.script_sig = script_sig
        self.sequence = sequence    # Sequence number

    @classmethod
    def parse(cls , s):
        from ALL_Class.Script import Script
        """Take a byte stream and parse the tx_input at the start , return a TxIn object"""
        # 1. 讀取 prev_tx
        # prev_tx is 32 bytes, little-endian
        prev_tx = s.read(32)[::-1]
        # 2. 讀取 prev_index
        # pre_index is an integer in 4 bytes, little-endian
        prev_index = little_endian_to_int(s.read(4))
        # 3. 讀取 script_sig
        # Use Script.parse to get the ScriptSig 
        script_sig = Script.parse(s)
        # 4. 讀取 sequence
        # sequence is an integer in 4 bytes, little-endian
        sequence = little_endian_to_int(s.read(4))
        
        # return an instance of TxIn 
        return cls(prev_tx, prev_index, script_sig, sequence) 
    
    def serialize(self):
        """Return the byte serialization of the transaction input """
        result = self.prev_tx[::-1]    # prev_tx is in little-endian
        result += int_to_little_endian(self.prev_index, 4)
        result += self.script_sig.serialize()
        result += int_to_little_endian(self.sequence, 4)

        return result
    
    def fetch_tx(self, testnet = False):
        from ALL_Class.TxFetcher import TxFetcher
        return TxFetcher.fetch(self.prev_tx.hex(), testnet=testnet)
    
    def value(self, testnet = False):
        """Return the value of the output being spent"""
        
        # 3. 取得交易物件
        tx = self.fetch_tx(testnet=testnet)
        # 4. 回傳金額
        return tx.tx_outs[self.prev_index].amount
    
    def script_pubkey(self, testnet = False):
        """Return the scriptPubKey of the output being spent"""
        # 3. 取得交易物件
        tx = self.fetch_tx(testnet=testnet)
        # 4. 回傳 scriptPubKey
        return tx.tx_outs[self.prev_index].script_pubkey
