import hashlib
import requests 
from io import BytesIO
from Bitcoin_S256Point import hash160
import logging
from op import * 

LOGGER = logging.getLogger(__name__)

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
    def parse (cls,s , testnet = False):
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
        version = int.from_bytes(s.read(4), 'little')
        # 2. 讀取輸入數量
        num_inputs = read_varint(s)
        # 3. 讀取輸入資料
        inputs = []
        for _ in range(num_inputs):
            inputs.append(TxIn.parse(s))
        # 4. 讀取輸出數量
        num_outputs = read_varint(s)
        # 5. 讀取輸出資料
        outputs = []
        for _ in range(num_outputs):
            outputs.append(Txout.parse(s))

        # 6. 讀取鎖定時間
        # locktime is an integer in 4 bytes, little-endian
        locktime = int.from_bytes(s.read(4), 'little')
        
        return cls(version, inputs, outputs, locktime, testnet = testnet)
    
    def serialize(self):
        """Return the byte serialization of the transaction"""
        result = self.version.to_bytes(4, 'little')
        
        result += encode_varint(len(self.tx_ins))
        for tx_in in self.tx_ins:
            result += tx_in.serialize()
        
        result += encode_varint(len(self.tx_outs))
        for tx_out in self.tx_outs:
            result += tx_out.serialize()
        
        result += self.locktime.to_bytes(4, 'little')

        return result
    def fee(self, testnet = False):
        """Return the transaction fee in satoshis"""
        # 1. 取得輸入金額
        input_amount = sum([tx_in.value(testnet) for tx_in in self.tx_ins])
        # 2. 取得輸出金額
        output_amount = sum([tx_out.amount for tx_out in self.tx_outs])
        # 3. 計算手續費
        return input_amount - output_amount


def read_varint(s):
    '''Read a variable-length integer from the stream'''
    # 1. 讀取第一個字元
    first_byte = s.read(1)
    # 2. 判斷長度
    if first_byte == b'\xfd':
        # 0xfd, 代表後面有兩個位元組
        return int.from_bytes(s.read(2), 'little')
    elif first_byte == b'\xfe':
        # 0xfe, 代表後面有四個位元組
        return int.from_bytes(s.read(4), 'little')
    elif first_byte == b'\xff':
        # 0xff, 代表後面有八個位元組
        return int.from_bytes(s.read(8), 'little')
    else:
        return int.from_bytes(first_byte, 'little')
    
def encode_varint(n):
    '''Encode a variable-length integer'''
    if n < 0xfd:
        return n.to_bytes(1, 'little')
    elif n <= 0xffff:
        return b'\xfd' + n.to_bytes(2, 'little')
    elif n <= 0xffffffff:
        return b'\xfe' + n.to_bytes(4, 'little')
    elif n <= 0xffffffffffffffff:
        return b'\xff' + n.to_bytes(8, 'little')
    else:
        return ValueError('integer too large:{}'.format(n))

def hash256(s):
    '''two round of sha256'''
    return hashlib.sha256(hashlib.sha256(s).digest()).digest()

class TxIn:
    def __init__(self, txid, vout, script_sig, sequence):
        self.txid = txid    # Previous transaction ID
        self.vout = vout    # Output index of the previous transaction
        if script_sig is None:
            self.script_sig = Script()
        else:
            self.script_sig = script_sig
        self.sequence = sequence    # Sequence number

    @classmethod
    def parse(cls , s):
        """Take a byte stream and parse the tx_input at the start , return a TxIn object"""
        # 1. 讀取 txid
        # prev_tx is 32 bytes, little-endian
        prev_tx = s.read(32)[::-1]
        # 2. 讀取 vout
        # pre_index is an integer in 4 bytes, little-endian
        prev_index = int.from_bytes(s.read(4), 'little')
        # 3. 讀取 script_sig
        # Use Script.parse to get the ScriptSig 
        script_sig = Script.parse(s)
        # 4. 讀取 sequence
        # sequence is an integer in 4 bytes, little-endian
        sequence = int.from_bytes(s.read(4), 'little')
        
        # return an instance of TxIn 
        return cls(prev_tx, prev_index, script_sig, sequence) 
    
    def serialize(self):
        """Return the byte serialization of the transaction input """
        result = self.txid[::-1]    # txid is in little-endian
        result += self.vout.to_bytes(4, 'little')
        result += self.script_sig.serialize()
        result += self.sequence.to_bytes(4, 'little')

        return result
    
    def fetch_tx(self, testnet = False):
        return TxFetcher.fetch(self.txid.hex(), testnet=testnet)
    
    def value(self, testnet = False):
        """Return the value of the output being spent"""
        # 1. 取得 txid
        # txid is in little-endian
        txid = self.txid.hex()
        # 2. 取得 vout
        vout = self.vout
        # 3. 取得交易物件
        tx = self.fetch_tx(testnet)
        # 4. 回傳金額
        return tx.tx_outs[vout].amount
    
    def script_pubkey(self, testnet = False):
        """Return the scriptPubKey of the output being spent"""
        # 1. 取得 txid
        # txid is in little-endian
        txid = self.txid.hex()
        # 2. 取得 vout
        vout = self.vout
        # 3. 取得交易物件
        tx = self.fetch_tx(testnet)
        # 4. 回傳 scriptPubKey
        return tx.tx_outs[vout].script_pubkey
    

class Txout: 
    def __init__(self, amount, script_pubkey):
        self.amount = amount    # Amount in satoshis
        self.script_pubkey = script_pubkey    # ScriptPubKey
    
    @classmethod
    def parse(cls, s):
        """Take a byte stream and parse the tx_output at the start , return a TxOut object"""
        # 1. 讀取 amount
        # amount is an integer in 8 bytes, little-endian
        amount = int.from_bytes(s.read(8), 'little')
        # 2. 讀取 script_pubkey
        # Use Script.parse to get the ScriptPubKey 
        script_pubkey = Script.parse(s)
        
        # return an instance of TxOut 
        return cls(amount, script_pubkey)
    def serialize(self):
        """ Return the byte serialization of the transaction output """

        result = self.amount.to_bytes(8, 'little')
        result += self.script_pubkey.serialize()

        return result
    

class TxFetcher:
    cathe = {}
    @classmethod
    def get_url(cls, testnet = False):
        if testnet :
            return "https://blockstream.info/testnet/api/"
        else:
            return "https://blockstream.info/api/"
    
    @classmethod
    def fetch(cls , tx_id, testnet = False , fresh = False):
        if fresh or (tx_id not in cls.cathe):
            url = "{}/tx/hex".format(cls.get_url(testnet), tx_id)
            response = requests.get(url)
            try:
                raw = bytes.fromhex(response.text.strip())
            except:
                raise ValueError('unexpected response: {}'.format(response.text))
            
        if raw[4] == 0:
            raw = raw[:4]+raw[6:]
            tx = Tx.parse(BytesIO(raw), testnet = testnet)
            tx.locktime = int.from_bytes(raw[-4:], 'little')
        else:
            tx = Tx.prase(BytesIO(raw), testnet = testnet)
        cls.cathe[tx_id] = tx
        return cls.cathe[tx_id]
    
  
class Script:
    def __init__(self, cmds = False):
        self.cmds = cmds if cmds else []

    @classmethod
    def parse(cls, s):
        """Take a byte stream and parse the script at the start , return a Script object"""
        # 1. 讀取 length
        # length is an integer in 1 byte
        length = read_varint(s)
        # 2. 讀取 cmds
        # cmds is a list of bytes
        count = 0  
        cmds = []
        while count < length:
            current = s.read(1)
            count += 1 
            current_byte = current[0]
            if current_byte >= 1 and current_byte <= 75:
                # 1~75 bytes
                cmds.append(s.read(current_byte))
                count += current_byte
            elif current_byte == 76:
                data_length = int.from_bytes(s.read(1), 'little')
                cmds.append(s.read(data_length))
                count += data_length + 1
            elif current_byte == 77:
                data_length = int.from_bytes(s.read(2), 'little')
                cmds.append(s.read(data_length))
                count += data_length + 2
            else:
                op_code = current_byte
                cmds.append(op_code)
        if count != length:
            raise SyntaxError('script length mismatch')
        # return an instance of Script 
        return cls(cmds)
    
    def raw_serialize(self):
        result = b''
        for cmd in self.cmds:
            if type(cmd) == int:
                result += int(cmd).to_bytes(1, 'little')
            else:
                length = len(cmd)
                if length < 75:
                    result += length.to_bytes(1, 'little')
                elif length > 75 and length < 256:
                    result += b'\x76' + length.to_bytes(1, 'little')
                elif length > 255 and length < 520:
                    result += b'\x77' + length.to_bytes(2, 'little')
                else:
                    raise ValueError('script length too long')
                result += cmd
        return result
    def serialize(self):
        result = self.raw_serialize()
        total = len(result)
        return encode_varint(total) + result
    
    def __add__(self,other):
        """Add two scripts together"""
        if type(other) == Script:
            return Script(self.cmds + other.cmds)
        else:
            raise TypeError('other must be a Script')
    
    def evaluate(self , z : list):
        cmds = self.cmds[:]
        stack = []
        altstack = []
        while len(cmds) > 0:
            cmd = cmds.pop(0)
            if type(cmd) == int:
                operation = OP_CODE_FUNCTIONS[cmd]
                # 1. 判斷是否為 op_code
                if cmd in (99,100):
                    if not operation(stack,cmds):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                elif cmd in (107,108):
                    if not operation(stack,altstack):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                elif cmd in (172,173,174,175):
                    if not operation(stack,z):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
                else:
                    if not operation(stack):
                        LOGGER.info("bad op: {}".format(OP_CODE_NAMES[cmd]))
                        return False
            else:
                # 2. 判斷是否為資料
                stack.append(cmd)
        # 3. 判斷 stack 是否為空
        if len(stack) == 0:
            return False 
        if stack.pop == b'':
            return False
        return True
    
    def __repr__(self):
        result = []
        for cmd in self.cmds:
            if type(cmd) == int:
                if OP_CODE_NAMES.get(cmd):
                    name = OP_CODE_NAMES.get(cmd)
                else:
                    name = 'OP_[{}]'.format(cmd)
                result.append(name)
            else:
                result.append(cmd.hex())
        return ' '.join(result)


