from io import BytesIO
from ALL_Class.Helper import *
from ALL_Class.op import * 
from ALL_Class.Module import * 

SIGHASH_ALL = 1  # SIGHASH_ALL is a constant for the signature hash type
SIGHASH_NONE = 2 
SIGNASH_SINGLE = 3
SIGHASH_ANTONECANPAY = 0x80

class Tx:
    def __init__ (self, version , tx_ins , tx_outs , locktime, testnet = False):
        self.version = version 
        self.tx_ins = tx_ins 
        self.tx_outs = tx_outs 
        self.locktime = locktime
        self.testnet = testnet 
    

    def hash(self):
        '''Binary hash of the legacy serialization'''
        return hash256(self.serialize())[::-1]
    
    def id(self):
        '''Human-readable hexadecimal of the transaction hash'''
        return self.hash().hex()    
    
    # cls: class parameter for class Tx, to set up class object
    # Serialization will return a new instance of a Tx object. So, it
    # needs to be called on the class itself (Tx.parse(...))
    # ■ 根據序列化資料，產生一個新的 Tx（交易）物件，讓方法可直接透過
    # Tx.parse(...) 呼叫，並使用 cls 來建立新實例
    # tx_obj = Tx.parse(serialization)
    @classmethod
    def parse (cls, stream , testnet = False):
        from ALL_Class.TxInput import TxIn
        from ALL_Class.TxOutput import TxOut
        '''Parse a transaction from a binary serialization'''
        
        """
        (2) Assume that the variable serialization is a byte string (bytes object)
                ■ Grab the first 4 bytes of the serialized data, which is the version field of a Bitcoin transaction
                ■ Stored in little-endian format
                    □ E.g., serialization = 01 00 00 00 01…
                    □ First 4 bytes: 01 00 00 00
                    □ version = 01000000 (little-endian integer)
        """
        # 1. 讀取版本號
        version = little_endian_to_int(stream.read(4))
        
        # 2. 讀取輸入數量
        num_inputs = read_varint(stream)
        # 3. 讀取輸入資料
        inputs = []
        for _ in range(num_inputs):
            inputs.append(TxIn.parse(stream))
        # 4. 讀取輸出數量
        num_outputs = read_varint(stream)
        # 5. 讀取輸出資料
        outputs = []
        for _ in range(num_outputs):
            outputs.append(TxOut.parse(stream))

        # 6. 讀取鎖定時間
        # locktime is an integer in 4 bytes, little-endian
        locktime = little_endian_to_int(stream.read(4))
        
        return cls(version, inputs, outputs, locktime, testnet = testnet)
    
    def serialize(self):
        """Return the byte serialization of the transaction"""
        result = int_to_little_endian(self.version, 4)
        
        result += encode_varint(len(self.tx_ins))
        for tx_in in self.tx_ins:
            result += tx_in.serialize()
        
        result += encode_varint(len(self.tx_outs))
        for tx_out in self.tx_outs:
            result += tx_out.serialize()
        
        result += int_to_little_endian(self.locktime, 4)

        return result
    def fee(self, testnet = False):
        """Return the transaction fee in satoshis"""
        # 1. 取得輸入金額
        input_amount = sum([tx_in.value(testnet=testnet) for tx_in in self.tx_ins])
        # 2. 取得輸出金額
        output_amount = sum([tx_out.amount for tx_out in self.tx_outs])
        # 3. 計算手續費
        return input_amount - output_amount
    
    def sig_hash(self, input_index , redeem_script = None ):
        from ALL_Class.TxInput import TxIn
        """
            □ Review transaction fields: Version, # of inputs, inputs, # of outputs, outputs, and locktime
            □ s = int_to_little_endian(self.version, 4):
                □ Start a byte‐array s with the 4-byte little-endian version field
            □ s += encode_varint(len(self.tx_ins)):
                □ Append the number of inputs (varint-encoded)
        """
        s = int_to_little_endian(self.version, 4)
        s += encode_varint(len(self.tx_ins))

        """
            □ Serialize every input:
                □ Each temp TxIn is serialized and appended to s
                □ if i == input_index:
                    ■ Replace its ScriptSig with the corresponding ScriptPubKey (standard rule for legacy signing)
                    ■ 將此輸入的 ScriptSig 改成對應 UTXO 的 ScriptPubKey
                □ else:
                    ■ leave ScriptSig empty
        """
        for i, tx_in in enumerate(self.tx_ins):
            if i == input_index:
                if redeem_script:
                    script_sig = redeem_script
                else:
                    script_sig = tx_in.script_pubkey(testnet = self.testnet)
            else:
                script_sig = None 
            s += TxIn(tx_in.prev_tx, tx_in.prev_index, script_sig, tx_in.sequence).serialize()

        """
            □ s += encode_varint(len(self.tx_outs))
                □ Append the number of outputs (varint)
            □ for tx_out in self.tx_outs:
                □ s += tx_out.serialize()
                □ Serialize every output (8-byte amount + ScriptPubKey) and append
        """
        s += encode_varint(len(self.tx_outs))
        for tx_out in self.tx_outs:
            s += tx_out.serialize()

        """
            □ Review fields:
                Version, # of inputs, inputs, # of outputs, outputs, and locktime
            □ s += int_to_little_endian(self.locktime, 4)
                □ Append the 4-byte locktime
            □ s += int_to_little_endian(SIGHASH_ALL, 4)
                □ Append the hash-type (here SIGHASH_ALL == 1) as 4 little-endian bytes         
        """
        s += int_to_little_endian(self.locktime, 4)
        s += int_to_little_endian(SIGHASH_ALL, 4)

        """
            □ h256 = hash256(s)
                □ Compute double SHA256 of the entire serialization
            □ return int.from_bytes(h256, 'big’)
                □ Return the 32-byte digest as a big-endian integer
        """
        h256 = hash256(s)
        return int.from_bytes(h256, 'big')
    
    def verify_input(self, input_index):
        from ALL_Class.Script import Script
        """
            ■ Use the TxIn.script_pubkey, Tx.sig_hash, and Script.evaluate methods
            ■ tx_in = self.tx_ins[input_index]
                ■ Pick the specific input we need to verify
            ■ script_pubkey = tx_in.script_pubkey(…)
                ■ Look up (via the referenced TXID / VOUT) the UTXO being spent and pull out its ScriptPubKey
        """
        tx_in = self.tx_ins[input_index] 
        script_pubkey = tx_in.script_pubkey(testnet = self.testnet)
        if script_pubkey.is_p2sh_script_pubkey():
            cmd = tx_in.script_sig.cmds[-1]
            raw_redeem = encode_varint(len(cmd)) + cmd
            redeem_script = Script.parse(BytesIO(raw_redeem))
        else:
            redeem_script = None
        """
            ■ z = self.sig_hash(input_index)
                ■ Calculate the signature-hash (𝑧) for this input
            ■ combined = tx_in.script_sig + script_pubkey
                ■ Combine the unlocking script (ScriptSig) provided in the current input with the locking script (ScriptPubKey) retrieved before
            ■ return combined.evaluate(z)
                ■ Evaluate the combined script
        """
        z = self.sig_hash(input_index , redeem_script)
        combined = tx_in.script_sig + script_pubkey
        return combined.evaluate(z)
    
    def verify(self):
        """ Verify the transaction """
        """
            1. Self.fee() < 0
                □ Make sure that we are not creating money
            2.if not self.verify_input(i):
                □ Make sure that each input has a correct ScriptSig
        """
        if self.fee() < 0 :
            return False 
        for i in range(len(self.tx_ins)):
            if not self.verify_input(i):
                return False
        return True
    def sign_input(self, input_index, priv_key):
        from ALL_Class.Script import Script
        """
            □ z = self.sig_hash(input_index)
                □ Compute the signature hash z for this input(SIGHASH_ALL)
            □ der = private_key.sign(z).der()
                □ Generate an ECDSA signature over z and encode it in DER
            □ sig = der + SIGHASH_ALL.to_bytes(1, 'big')
                □ Append the 1-byte sighash-type (0x01) to obtain the signature
            □ sec = private_key.point.sec()
                □ Export the signer’s public key in SEC format
            □ self.tx_ins[input_index].script_sig = Script([sig, sec])
                □ Build a P2PKH-style ScriptSig([signature, pubkey]) and assign it to the chosen input
            □ return self.verify_input(input_index)
                □ Run the combined script to confirm the signature is valid; returns True or False

        """
        z = self.sig_hash(input_index)
        der = priv_key.sign(z).der()
        sig = der + SIGHASH_ALL.to_bytes(1, 'big')
        sec = priv_key.point.sec()
        self.tx_ins[input_index].script_sig = Script([sig, sec])
        
        return self.verify_input(input_index)
