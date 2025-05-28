from ALL_Class.Helper import *
from ALL_Class.op import * 
from ALL_Class.Module import * 



class TxOut: 
    def __init__(self, amount, script_pubkey):
        self.amount = amount    # Amount in satoshis
        self.script_pubkey = script_pubkey    # ScriptPubKey
    
    @classmethod
    def parse(cls, s):
        from ALL_Class.Script import Script
        """Take a byte stream and parse the tx_output at the start , return a TxOut object"""
        # 1. 讀取 amount
        # amount is an integer in 8 bytes, little-endian
        amount = little_endian_to_int(s.read(8))
        # 2. 讀取 script_pubkey
        # Use Script.parse to get the ScriptPubKey 
        script_pubkey = Script.parse(s)
        
        # return an instance of TxOut 
        return cls(amount, script_pubkey)
    def serialize(self):
        """ Return the byte serialization of the transaction output """

        result = int_to_little_endian(self.amount, 8)
        result += self.script_pubkey.serialize()

        return result
